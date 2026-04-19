'''
Reduces code duplication across pipeline.stage_two and evaluation.harness subprocesses.
Enables easy import and use of config schema & loader/s in both settings.
'''

from datetime import datetime, timezone
from argparse import Namespace
import sys

from logmap_llm.config.loader import load_and_validate_config
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.utils.logging import TeeWriter, info

def subprocess_bootstrap(subprocess_name: str, args: Namespace):
    """
    Shared bootstrapping for stage_two.py and evaluation/harness.py subprocesses
    """
    cfg = load_and_validate_config(
        args.config,
        reuse_align=getattr(args, 'reuse_align', False),
        reuse_prompts=getattr(args, 'reuse_prompts', False),
        reporting=False,
    )
    run_paths = PipelinePaths.from_config(cfg)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = run_paths.subprocess_log(timestamp=timestamp, subprocess_name=subprocess_name)

    original_stdout = sys.stdout
    tee = TeeWriter(str(log_path), original_stdout)
    sys.stdout = tee

    info(f"({subprocess_name}) Pipeline log file: {log_path}")

    return cfg, run_paths, tee
