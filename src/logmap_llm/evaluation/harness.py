"""
logmap_llm.evaluation.harness
"""
from __future__ import annotations
 
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any
 
from logmap_llm.evaluation.engines import (
    CustomEvaluationEngine,
    DeepOntoEvaluationEngine,
    EvaluationEngine,
    PartialReferenceEvaluationEngine,
)
from logmap_llm.evaluation.io import (
    load_mapping_pairs,
    load_oracle_predictions,
)
from logmap_llm.utils.logging import (
    metric
)


def select_engine(partial_reference: bool = False, force_custom: bool = False) -> EvaluationEngine:
    """
    Selects an evaluation engine for the evaluate stage.
    The choice is between PartialReference, Custom and DeepOnto
    evaluation engines. The thinking is essentially:
        1. Are we dealing with a KG-like task? => PartialReference
        2. CustomEvaluationEngine is fine in most other cases, unless:
        3. We want to verify our results from the custom implementation
           agaisnt a trusted evaluator... OR: we want to evaluate subsumption
           based ranking for BioML (in both cases, we need DeepOnto).
    However, note that the path that scores subsumption-based ranking is
    not yet implemented. The logic swicthes based on values provided within
    the config.toml that is provided to the system, under [evaluate]
        partial_reference: bool, force_custom: bool; else: DeepOnto
    """
    if partial_reference:
        return PartialReferenceEvaluationEngine()
    if force_custom:
        return CustomEvaluationEngine()
    if DeepOntoEvaluationEngine.is_available():
        return DeepOntoEvaluationEngine()
    return CustomEvaluationEngine()


###
# HELPERS
#  (for printing)
###

_BREAKLINE_ON_ORACLE_METRIC_KEYS = frozenset({"Sensitivity"})

def _format_metric(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    # else:
    return value


def _format_metric_key(metric_key: str) -> str:
    return (" ".join(metric_key.split("_"))).title()


def _print_global_metrics(metrics: dict) -> None:
    metric(f"Global Alignment Metrics:")
    metric(f"  running metrics evaluation backend: {metrics.get('source', 'unknown')})")
    metric(f"  ")
    equal_spacing = max((len(k) for k in metrics), default=0)
    for metric_key, metric_value in metrics.items():
        metric(f"   {_format_metric_key(metric_key):<{equal_spacing}}  :  {_format_metric(metric_value)}")


def _print_oracle_metrics(metrics: dict, ignore_keys: list[str] = ['false_mappings']) -> None:
    metric(f"Oracle Discrimination Metrics")
    metric(f"  running metrics evaluation backend: {metrics.get('source', 'unknown')})")
    metric(f"  ")
    reversed_metrics_dict: dict = dict(reversed(metrics.items())) # purely cosmetric
    equal_spacing = max((len(k) for k in reversed_metrics_dict), default=0)
    for metric_key, metric_value in reversed_metrics_dict.items():
        if metric_key not in ignore_keys:
            metric(f"   {_format_metric_key(metric_key):<{equal_spacing}}  :  {_format_metric(metric_value)}")
            if metric_key in _BREAKLINE_ON_ORACLE_METRIC_KEYS: print() # noqa


def _print_stratified_results(strat: dict, label: str) -> None:
    print(f"\n{label}:")
    for key, m in strat.items():
        extra = f", Ign={m['ignored']}" if "ignored" in m else ""
        metric(
            f"  {key:>15s}:  P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
            f"(TP={m['true_positives']}, "
            f"FP={m['false_positives']}, "
            f"FN={m['false_negatives']}{extra})"
        )


###
# HELPERS FOR FILE OPERATIONS
###


_FALSE_MAPPING_FIELDS = [
    "source_entity_uri",
    "target_entity_uri",
    "oracle_prediction",
    "oracle_confidence",
    "in_reference",
    "error_type",
]


def _write_false_mappings_csv(false_mappings: list[dict], output_json_path: Path | None, oracle_predictions_path: Path) -> Path:
    """Write a false-mappings CSV alongside the evaluation results"""
    # TODO: handle this via the paths.py -- at present, this is fragile
    if output_json_path is not None:
        fm_path = Path(output_json_path).parent / "false_mappings.csv"
    else:
        fm_path = Path(oracle_predictions_path).parent / "false_mappings.csv"
 
    fm_path.parent.mkdir(parents=True, exist_ok=True)

    with open(fm_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=_FALSE_MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(false_mappings)

    return fm_path
 
 
###
# HELPERS FOR AUTOMATIC FILE DETECTION
# TODO: retire these operations as legacy when the process orchestration
# layer is fleshed out; for now, we can retain these (as they can be quite
# helpful for ad-hoc & quick testing operations; but they're coupled to 
# any naming conventions for our specific experimental setting).
# Also, they should be using the paths.py module (as mentioned: legacy code)
###
 
def _find_system_mappings(results_dir: Path, task_name: str) -> Path | None:
    refined_dir = Path(results_dir) / "logmap-refined-alignment"
    if not refined_dir.is_dir():
        return None
    tsv = refined_dir / f"{task_name}-logmap_mappings.tsv"
    if tsv.exists():
        return tsv
    for f in refined_dir.glob("*mappings.tsv"):
        return f
    return None
 
def _find_oracle_predictions(results_dir: Path, task_name: str) -> Path | None:
    output_dir = Path(results_dir) / "logmapllm-outputs"
    if not output_dir.is_dir():
        return None
    for f in output_dir.glob("*predictions*.csv"):
        return f
    return None
 
def _find_reference_alignment(datasets_dir: Path, task_name: str) -> Path | None:
    datasets_dir = Path(datasets_dir)
    candidates: list[Path] = []
    if task_name == "anatomy":
        candidates.append(datasets_dir / "anatomy" / "refs_equiv" / "full.tsv")
    elif task_name.startswith("conference-"):
        pair = task_name.split("-", 1)[1]
        candidates.append(datasets_dir / "conference" / "refs_equiv" / f"{pair}.tsv")
    elif task_name.startswith("kg-"):
        pair = task_name[3:]
        candidates.append(
            datasets_dir / "knowledgegraph" / pair / "refs_equiv" / "reference_all.tsv"
        )
    else:
        candidates.append(datasets_dir / "bio-ml" / task_name / "refs_equiv" / "full.tsv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
 

###
# EVALUATION ORCHESTRATION
# ------------------------
# LOCAL PRIVATE CONSTANTS:
#   _VALID_METRICS:
#       1. 'global': calculates the global alignment metrics (P/R/F1)
#           alongside TP/FP/FN, system alignment size (ie. every mapping
#           that was accepted by LogMap, both through its own alignment
#           system, as well as after having refined the oracle mappings)
#           in addition to the reference alignment size (the number of 
#           mappings present in the reference alignment) and the source
#           (the source is the evaluation backend / method eg. custom
#            deeponto or partial reference/kg_partial)
#       2. 'oracle': calculates the oracle metrics (P/R/F1/YI/Sen/Spec)
#           it also includes the counts for the 'total candidates'
#           which is (typically) the M_ask size; the 'oracle excluded'
#           number, which represents mappings that could not be answered
#           as well as 'partial scope excluded' (when using partial 
#           reference alignments) and the number of errors + TP/TN/FP/FN
#       3. 'ranking' -- this is currently a stub ready for BioML-based 
#           ranking metrics.
#
#       _DEFAULT_METRICS:
#           simply: which of the valid metrics do we use by default?
#           they're specified here.
###

_VALID_METRICS = {"global", "ranking", "oracle"}
_DEFAULT_METRICS = ["global", "oracle"]


def evaluate_alignment(
    system_mappings_path: Path,
    reference_path: Path,
    oracle_predictions_path: Path | None = None,
    train_reference_path: Path | None = None,
    test_cands_path: Path | None = None,
    initial_alignment_path: Path | None = None,
    task_name: str = "",
    metrics: list[str] | None = None,
    output_json_path: Path | None = None,
    force_custom: bool = False,
    partial_reference: bool = False,
    stratified_by_entity_type: bool = False,
    stratified_class_property: bool = False,
) -> dict:
    """
    Run full evaluation and return the results dict.
    """
    metrics = metrics or list(_DEFAULT_METRICS)
    system_mappings_path = Path(system_mappings_path)
    reference_path = Path(reference_path)
 
    # pick an engine for this task based on the config flags
    engine = select_engine(
        partial_reference=partial_reference,
        force_custom=force_custom,
    )
    results: dict[str, Any] = {
        "task_name": task_name,
        "engine": engine.name(),
    }
 
    if partial_reference:
        print("NOTE: Partial reference alignment (PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE semantics)")
 
    ###
    # GLOBAL MATCHING
    ###

    if "global" in metrics:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - -")
 
        global_metrics = engine.compute_global(
            system_mappings_path,
            reference_path,
            train_reference_path=(
                Path(train_reference_path) if train_reference_path else None
            ),
        )
        results["global"] = global_metrics
        _print_global_metrics(global_metrics)
 
        if stratified_by_entity_type and engine.supports("stratified_global"):
 
            strat_refs: dict[str, Path] = {}
            ref_dir = reference_path.parent
            for etype in ("class", "property", "instance"):
                p = ref_dir / f"reference_{etype}.tsv"
                if p.exists():
                    strat_refs[etype] = p
 
            strat_kwargs: dict[str, Any] = {}

            if strat_refs:
                strat_kwargs["stratified_refs"] = strat_refs
 
            strat = engine.compute_stratified_global(
                system_mappings_path,
                reference_path,
                **strat_kwargs,
            )

            if strat:
                label_mode = "explicit refs" if strat_refs else "URI pattern"
                gs_mode = "partial" if partial_reference else "complete"
                _print_stratified_results(
                    strat, f"Stratified evaluation ({label_mode}, {gs_mode} GS)",
                )
                for etype, m in strat.items():
                    results[f"global_{etype}"] = m
 
        if stratified_class_property:
            from logmap_llm.utils.misc import compute_conference_m1_m2_stratified
 
            m1_m2 = compute_conference_m1_m2_stratified(
                system_mappings_path=system_mappings_path,
                reference_path=reference_path,
                initial_alignment_path=(
                    Path(initial_alignment_path) if initial_alignment_path else None
                ),
            )

            if m1_m2:
                _print_stratified_results(m1_m2, "Conference M1/M2 stratified evaluation")
                for key, m in m1_m2.items():
                    results[f"global_{key}"] = m
 
    ###
    # RANKING METRICS
    ###
    
    if "ranking" in metrics:
        print("not yet implemented!")
 
    ###
    # ORACLE DISCRIMINATION METRICS
    ###

    if "oracle" in metrics:
        oracle_path = Path(oracle_predictions_path) if oracle_predictions_path else None
        if oracle_path is not None and oracle_path.exists():
            reference_pairs = load_mapping_pairs(reference_path)
            predictions = load_oracle_predictions(oracle_path)
 
            oracle_metrics = engine.compute_oracle(
                predictions,
                reference_pairs,
            )

            results["oracle"] = oracle_metrics
 
            print("\n- - - - - - - - - - - - - - - - - - - - - - - -")
            _print_oracle_metrics(oracle_metrics)
 
            false_mappings = oracle_metrics.get("false_mappings", [])
            if false_mappings:
                fm_path = _write_false_mappings_csv(
                    false_mappings,
                    output_json_path,
                    oracle_path,
                )
                print(f"\nFalse mappings ({len(false_mappings)} FP+FN) saved to: {fm_path}")
        else:
            print("\nOracle predictions not found — skipping oracle metrics.")
 
    ###
    # WRITE JSON
    ###

    if output_json_path is not None:
        serialisable = {
            k: (
                {kk: vv for kk, vv in v.items() if kk != "false_mappings"}
                if isinstance(v, dict) and "false_mappings" in v
                else v
            )
            for k, v in results.items()
        }
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        print(f"\nEvaluation results saved to: {output_json_path}")
 
    return results


 
def _build_arg_parser() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(description="LogMap-LLM Evaluation")
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--system", type=str, required=False)
    parser.add_argument("--reference", type=str, required=False)
    parser.add_argument("--oracle-predictions", type=str, default=None)
    parser.add_argument("--initial-alignment", type=str, default=None)
    parser.add_argument("--train-reference", type=str, default=None)
    parser.add_argument("--test-cands", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="global,oracle")
    parser.add_argument("--partial-reference", action="store_true", default=False)
    parser.add_argument("--no-deeponto", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None)
    return parser
 
 



def _run_from_config(args: argparse.Namespace) -> dict:
    """
    canonical mode for invoking; loads and validate the TOML config via the shared
    subprocess bootstrap (in logmap_llm.utils.subprocess) and derive paths through 
    PipelinePaths (single source of truth, shared with the pipeline runner).
    Dispatches to evaluate_alignment.
    """
    from logmap_llm.utils.subprocess import subprocess_bootstrap

    cfg, run_paths, _tee = subprocess_bootstrap("EVALUATE", args=args)
    eval_cfg = cfg.evaluation

    force_custom = args.no_deeponto or eval_cfg.force_custom_eval
    partial_ref = args.partial_reference or eval_cfg.partial_reference

    system_path = run_paths.refined_mappings_tsv()
    oracle_path = run_paths.predictions_csv()
    initial_align_path = run_paths.logmap_m_ask()
    
    output_path = Path(args.output) if args.output else run_paths.eval_json()

    metrics = [metric.strip() for metric in args.metrics.split(",")]

    return evaluate_alignment(
        system_mappings_path=system_path,
        reference_path=eval_cfg.reference_alignment_path or "",
        oracle_predictions_path=oracle_path if oracle_path.exists() else None,
        train_reference_path=eval_cfg.train_alignment_path,
        test_cands_path=eval_cfg.test_cands_path,
        initial_alignment_path=initial_align_path if initial_align_path.exists() else None,
        task_name=cfg.alignmentTask.task_name,
        metrics=metrics,
        output_json_path=output_path,
        force_custom=force_custom,
        partial_reference=partial_ref,
        stratified_by_entity_type=eval_cfg.stratified_by_entity_type,
        stratified_class_property=eval_cfg.stratified_class_property,
    )



def _run_from_explicit_paths(args: argparse.Namespace) -> dict:
    """
    Secondlry invocation mode, where all paths passed explicitly by/via the 
    command line; this bypasses config loading and the bootstrap helper entirely
    which results in no cfg object and no subprocess log (but allows more easily
    testable code -- TODO test suite)
    """
    metrics = [metric.strip() for metric in args.metrics.split(",")]
    return evaluate_alignment(
        system_mappings_path=args.system,
        reference_path=args.reference,
        oracle_predictions_path=args.oracle_predictions,
        train_reference_path=args.train_reference,
        test_cands_path=args.test_cands,
        initial_alignment_path=args.initial_alignment,
        task_name=args.task_name,
        metrics=metrics,
        output_json_path=args.output,
        force_custom=args.no_deeponto,
        partial_reference=args.partial_reference,
    )



def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    if args.config:
        results = _run_from_config(args)
    else:
        results = _run_from_explicit_paths(args)

    summary = {
        key: value
        for key, value in results.items()
        if not (isinstance(value, dict) and "false_mappings" in value)
    }

    print(json.dumps(summary, indent=2, default=str))
 


if __name__ == "__main__":
    main()
 