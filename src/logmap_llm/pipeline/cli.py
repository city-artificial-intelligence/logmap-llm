"""
logmap_llm.pipeline.cli — command-line argument parsing for pipeline
Provides the ArgumentParser for the LogMap-LLM pipeline entry point.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    
    parser = ArgumentParser(description="LogMap-LLM: Python-based LogMap LLM extension")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default_config.toml",
        help="Path to the TOML configuration file (default: configs/default_config.toml)",
    )
    parser.add_argument(
        "--reuse-align",
        action="store_true",
        default=False,
        help="Override config to reuse existing LogMap alignment (skips initial alignment)",
    )
    parser.add_argument(
        "--reuse-prompts",
        action="store_true",
        default=False,
        help=(
            "Override config to reuse existing prompts (skips init alignment + building prompts; "
            "implies --reuse-align)"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable owlready2 quadstore caching (parse ontologies from scratch) (TODO)",
    )

    return parser.parse_args()
