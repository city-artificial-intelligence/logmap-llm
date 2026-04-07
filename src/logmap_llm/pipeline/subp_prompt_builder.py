'''
logmap_llm.pipeline.stage_two — subprocess prompt builder

This module runs as a subprocess with `python -m logmap_llm.pipeline.stage_two`
and isolates owlready2 from JPype.  Note that the TOML config is read directly via
`tomlib` — it should not import from ``logmap_llm.config.loader`` (or any other
module that transitively imports JPype).
'''

from __future__ import annotations

import argparse
import os
import sys
import json
import pandas as pd
import tomllib

from logmap_llm.log_utils import error, warning, info, success
from logmap_llm.constants import (
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    COL_RELATION,
    COL_CONFIDENCE,
    COL_ENTITY_TYPE,
    M_ASK_COLUMNS,
    PAIRS_SEPARATOR,
)


# copied from `constants.py`
def get_m_ask_column_names() -> list[str]:
    """
    Returns M_ASK_COLUMNS as a mutable list
    """
    return list(M_ASK_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="LogMap-LLM: subprocess prompt builder")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the TOML configuration file",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        choices=["conference", "bioml", "anatomy", "knowledgegraph"],
        help="Track identifier (forwarded from parent process)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable owlready2 quadstore caching",
    )
    args = parser.parse_args()

    config_path = args.config

    if not os.path.isfile(config_path):
        error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, mode="rb") as fp:
        config = tomllib.load(fp)

    info(f"Stage 2 subprocess: config loaded from {config_path}")

    # resolve track
    track = args.track or config.get('alignmentTask', {}).get('track', None)
    if track:
        info(f"Track: {track}")

    task_name = config['alignmentTask']['task_name']
    onto_src_filepath = config['alignmentTask']['onto_source_filepath']
    onto_tgt_filepath = config['alignmentTask']['onto_target_filepath']
    logmap_outputs_dir_path = config['outputs']['logmap_initial_alignment_output_dirpath']

    # load M_ask from file
    info('Loading M_ask from initial alignment ...')
    filename = task_name + '-logmap_mappings_to_ask_oracle_user_llm.txt'
    filepath = os.path.join(logmap_outputs_dir_path, filename)

    # handle empty M_ask
    if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
        warning("M_ask file is empty or missing — no prompts to build.")
        sys.exit(0)

    m_ask_df = pd.read_csv(filepath, sep='|', header=None)
    m_ask_df.columns = get_m_ask_column_names()

    oupt_name = config['oracle']['oracle_user_prompt_template_name']

    info(f'Building fresh Oracle user prompts with template: {oupt_name}')

    # import prompt building (owlready2 runs here, in the subprocess)
    from logmap_llm.oracle.prompts.templates import build_oracle_user_prompts

    m_ask_oracle_user_prompts = build_oracle_user_prompts(
        oupt_name, onto_src_filepath, onto_tgt_filepath, m_ask_df
    )

    if m_ask_oracle_user_prompts is not None:
        info(f"Number of LLM Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}")

    # save prompts to JSON (allows parent process to read them)
    # TODO: probably should handle these via PipelinePaths obj?
    dirpath = config['outputs']['logmapllm_output_dirpath']
    os.makedirs(dirpath, exist_ok=True)
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_oracle_user_prompts.json'
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'w') as fp:
        json.dump(m_ask_oracle_user_prompts, fp)
    success(f'LLM Oracle user prompts saved to: {filename}')


if __name__ == "__main__":
    main()
