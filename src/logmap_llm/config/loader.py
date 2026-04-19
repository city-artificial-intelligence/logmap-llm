"""
logmap_llm.config.loader — Config loading, validation, and display utilities.
Loads TOML config, applies CLI overrides, validates via Pydantic schema.
"""
from __future__ import annotations

import sys
import os
import tomllib
from typing import Any

from logmap_llm.utils.logging import error, info, success
from logmap_llm.config.schema import validate_config, LogMapLLMConfig
from pydantic import ValidationError


def load_and_validate_config(
    config_path: str,
    reuse_align: bool = False,
    reuse_prompts: bool = False,
    reporting: bool = True,
) -> LogMapLLMConfig:
    """Load a TOML config file, apply CLI overrides, and validate"""
    if not os.path.isfile(config_path):
        error(f"configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, mode="rb") as fp:
        config = tomllib.load(fp)

    if reuse_align:
        info("--reuse-align set, overriding config (reuse init align)")
        config['pipeline']['align_ontologies'] = 'reuse'

    if reuse_prompts:
        info("--reuse-prompts set, overriding config (reuse prompts+align)")
        config['pipeline']['build_oracle_prompts'] = 'reuse'
        config['pipeline']['align_ontologies'] = 'reuse'

    try:
        cfg = validate_config(config)
    except ValidationError as e:
        error(f"configuration file is invalid: {config_path}")
        for err in e.errors():
            field = " x ".join(str(loc) for loc in err["loc"])
            error(f" {field}: {err['msg']}")
        sys.exit(1)

    if reporting:
        success(f"configuration validated: {config_path}")
    
    return cfg


def inspect_and_mask_api_key(key: str) -> str:
    """Mask an API key for display, showing only first and last 4 chars"""
    if key is None: # guards agaisnt None
        return "<unset>"
    if key == 'EMPTY':
        return key
    return f"{key[:4]} ... {key[-4:]}" if len(key) > 8 else "***"


def parse_config_into_list(
    config_dict: dict, key_prefix: str = ""
) -> list[tuple[str, Any]]:
    """Recursively flatten a nested config dict into (dotted-key, value) pairs."""
    if not isinstance(config_dict, dict):
        return [(key_prefix, config_dict)]
    if not config_dict and key_prefix:
        return [(key_prefix, {})] # antisipate empty dicts
    kv_config_pairs = []
    for key, value in config_dict.items():
        extended_key = f"{key_prefix}.{key}" if key_prefix else key
        kv_config_pairs.extend(parse_config_into_list(value, extended_key))
    return kv_config_pairs


def print_config_summary(cfg: LogMapLLMConfig) -> None:
    """Print a human-readable summary of the configuration."""
    flat_config_params: list = parse_config_into_list(cfg.model_dump())
    expr_params_str: str = "Summary of Experiment Parameters:\n\n"
    for key, value in flat_config_params:
        if key == "oracle.api_key":
            value = inspect_and_mask_api_key(value)
        elif value is None:
            value = "<unset>"
        expr_params_str += f"{key}: {value}\n"
    info(expr_params_str)
