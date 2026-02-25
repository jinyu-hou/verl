# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses as _dc
from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def _get_dataclass_defaults(dc_type) -> dict:
    """Recursively extract non-MISSING default values from a dataclass type.

    MISSING fields (required fields with no default) are skipped so that the
    returned dict can be used as a non-struct OmegaConf base without triggering
    MissingMandatoryValue errors.

    _target_ is also excluded: it is a Hydra convention and should only appear
    when explicitly set by the user config, not defaulted to '' from BaseConfig.
    """
    result = {}
    for f in _dc.fields(dc_type):
        # Skip Hydra's _target_ — an empty default would trigger instantiate('') later
        if f.name == "_target_":
            continue
        if f.default is not _dc.MISSING:
            val = f.default
        elif f.default_factory is not _dc.MISSING:
            try:
                val = f.default_factory()
            except Exception:
                continue
        else:
            continue  # Truly MISSING (mandatory) — skip
        if _dc.is_dataclass(val):
            result[f.name] = _get_dataclass_defaults(type(val))
        else:
            result[f.name] = val
    return result


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass instance.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: When provided, merge config on top of dataclass defaults,
            filter to known fields, and instantiate the dataclass (running __post_init__).
            Extra keys not in the dataclass schema are silently dropped, avoiding
            struct-mode rejection of project-specific keys (e.g. grad_norm_threshold).
            When None, the config must contain _target_ for Hydra instantiate, or is
            returned as-is for backward compatibility.

    Returns:
        A dataclass instance (when dataclass_type is given),
        or the Hydra-instantiated object / raw config (when dataclass_type is None).
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got a non-config object (e.g. already an HFModelConfig instance)
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        # Only route to Hydra instantiate when _target_ is present AND non-empty.
        # BaseConfig sets _target_='' as a default which must not trigger instantiate.
        _target = config.get("_target_", None) if isinstance(config, (DictConfig, dict)) else None
        if not _target:
            return config
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")

    # Convert the user config to a plain container to strip any struct-mode constraints,
    # then rebuild as a plain (non-struct) OmegaConf DictConfig.
    if isinstance(config, DictConfig | ListConfig):
        cfg_container = OmegaConf.to_container(config, resolve=False, throw_on_missing=False)
    else:
        cfg_container = config

    # Build non-struct defaults from the dataclass, skipping MISSING fields.
    # Using non-struct OmegaConf avoids:
    #   - struct-mode rejection of project-specific extra keys (e.g. grad_norm_threshold)
    #   - nested schema mismatches (e.g. FSDPOptimizerConfig fields in OptimizerConfig slot)
    #   - MissingMandatoryValue errors from nested configs (e.g. ProfilerConfig.tool)
    defaults = _get_dataclass_defaults(dataclass_type)
    cfg_merged = OmegaConf.merge(OmegaConf.create(defaults), OmegaConf.create(cfg_container))
    # Strip empty _target_ that may have leaked from BaseConfig defaults or user YAML
    # to prevent downstream omega_conf_to_dataclass(result) from routing to instantiate('').
    if isinstance(cfg_merged, DictConfig) and cfg_merged.get("_target_", None) == "":
        with open_dict(cfg_merged):
            del cfg_merged["_target_"]
    # Instantiate the dataclass with only the fields it declares, filtering out any
    # project-specific extra keys. This runs __post_init__ (e.g. HFModelConfig loads
    # hf_config, tokenizer, etc.) while still tolerating unknown keys in the YAML.
    known_fields = {f.name for f in _dc.fields(dataclass_type)}
    merged_container = OmegaConf.to_container(cfg_merged, resolve=True, throw_on_missing=False)
    init_kwargs = {k: v for k, v in merged_container.items() if k in known_fields}
    return dataclass_type(**init_kwargs)


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "reward_model": "micro_batch_size",
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    # Check for reward model micro-batch size conflicts
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        check_mutually_exclusive(
            config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
        )

    if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    if config.actor_rollout_ref.model.get("lora_rank", 0) > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        assert config.actor_rollout_ref.model.lora_rank <= 512, "LoRA rank in vLLM must be less than or equal to 512"

    print("[validate_config] All configuration checks passed successfully!")
