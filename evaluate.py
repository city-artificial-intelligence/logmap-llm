'''
LogMap-LLM Alignment Evaluator
-------------------------------

Computes evaluation metrics for LogMap-LLM alignment results. 
Uses DeepOnto (if installed) for official OAEI Bio-ML metrics, with custom fallback for environments without DeepOnto.

Three configurable metric groups:

  1. 'global' — Precision, Recall, F1:
     Full system output vs. reference alignment.
     Uses DeepOnto's AlignmentEvaluator.f1() when available,
     with support for null (training) mappings in semi-supervised setting.

  2. 'ranking' — MRR, Hits@K:
     Local ranking evaluation on candidate mappings.
     Requires DeepOnto and test.cands.tsv file.

  3. 'oracle' — Sensitivity, Specificity, Youden's J:
     Oracle discrimination on m_ask candidates.
     Always uses custom implementation (no DeepOnto equivalent).

Usage:

    # standalone with all metrics
    python evaluate.py \\
        --results-dir results/anatomy/model/template \\
        --reference refs_equiv/full.tsv \\
        --task-name anatomy \\
        --metrics global oracle

    # with ranking (requires test.cands.tsv)
    python evaluate.py \\
        --results-dir results/snomed-fma.body/model/template \\
        --reference refs_equiv/full.tsv \\
        --test-cands refs_equiv/test.cands.tsv \\
        --task-name snomed-fma.body \\
        --metrics global ranking oracle

    # semi-supervised (exclude training mappings)
    python evaluate.py \\
        --results-dir results/snomed-fma.body/model/template \\
        --reference refs_equiv/test.tsv \\
        --train-reference refs_equiv/train.tsv \\
        --task-name snomed-fma.body \\
        --metrics global oracle

    # skip DeepOnto (use custom evaluation only)
    python evaluate.py \\
        --results-dir results/anatomy/model/template \\
        --reference refs_equiv/full.tsv \\
        --task-name anatomy \\
        --no-deeponto \\
        --metrics global oracle

    # custom JVM memory for DeepOnto
    python evaluate.py \\
        --results-dir results/anatomy/model/template \\
        --reference refs_equiv/full.tsv \\
        --task-name anatomy \\
        --jvm-memory 16g \\
        --metrics global oracle
'''

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path


# ----------------------------------------------
# DeepOnto availability check (deferred import)
# ----------------------------------------------

# DeepOnto initialises OWLAPI via JPype on import, which:
#   (a) prompts for JVM memory unless JAVA_MEMORY env var is set
#   (b) starts a JVM that cannot be restarted in the same process
#
# We defer the import to first use so that CLI flags:
#  (--jvm-memory, --no-deeponto) 
# can take effect before the JVM starts

_DEEPONTO_AVAILABLE = None  # None = not yet checked
_DEEPONTO_DISABLED = False  # set True by --no-deeponto flag
_deeponto_modules = {}      # cache for imported modules


def _ensure_jvm_memory(memory: str = "8g"):
    """
    set all known DeepOnto JVM memory env vars
    DeepOnto has used different variable names across versions.
    We set all of them to ensure non-interactive operation.
    """
    for var_name in ("JAVA_MEMORY", "DEEPONTO_JVM_MEMORY", "JVM_MEMORY"):
        if var_name not in os.environ:
            os.environ[var_name] = memory


def _try_import_deeponto():
    """attempt to import DeepOnto, caching the result"""
    global _DEEPONTO_AVAILABLE, _deeponto_modules
    if _DEEPONTO_AVAILABLE is not None:
        return _DEEPONTO_AVAILABLE

    _ensure_jvm_memory()

    try:
        from deeponto.align.evaluation import AlignmentEvaluator
        from deeponto.align.mapping import ReferenceMapping, EntityMapping
        from deeponto.align.oaei import ranking_eval
        _deeponto_modules["AlignmentEvaluator"] = AlignmentEvaluator
        _deeponto_modules["ReferenceMapping"] = ReferenceMapping
        _deeponto_modules["EntityMapping"] = EntityMapping
        _deeponto_modules["ranking_eval"] = ranking_eval
        _DEEPONTO_AVAILABLE = True
    except ImportError:
        _DEEPONTO_AVAILABLE = False

    return _DEEPONTO_AVAILABLE


def deeponto_available() -> bool:
    if _DEEPONTO_DISABLED:
        return False
    return _try_import_deeponto()


# -----------------------------------------------------------
# Format conversion: LogMap output -> DeepOnto-compatible TSV
# -----------------------------------------------------------

def _detect_tsv_format(filepath: Path) -> str:
    """
    detect whether a TSV file has a DeepOnto header or uses OAEI format
    Returns:
        'deeponto' — has SrcEntity/TgtEntity/Score header
        'oaei'     — no header, 5-column OAEI format (uri, uri, =, score, CLS)
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
        if first_line.startswith("SrcEntity") or first_line.startswith("src"):
            return "deeponto"
        return "oaei"


def _convert_to_deeponto_tsv(input_path: Path, output_path: Path) -> None:
    """
    convert a LogMap/OAEI-format TSV to DeepOnto's expected format

    Input format (no header):  uri \\t uri \\t = \\t score \\t CLS
    Output format (with header): SrcEntity \\t TgtEntity \\t Score
    """
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        first_line = fin.readline().strip()
        # if already has DeepOnto header, just copy it
        if first_line.startswith("SrcEntity") or first_line.startswith("src"):
            fout.write(first_line + "\n")
            for line in fin:
                fout.write(line)
            return

        # write header
        fout.write("SrcEntity\tTgtEntity\tScore\n")

        # process the first line
        if first_line:
            parts = first_line.split("\t")
            if len(parts) >= 2:
                src = parts[0].strip()
                tgt = parts[1].strip()
                score = "1.0"
                if len(parts) >= 4:
                    try:
                        score = str(float(parts[3].strip()))
                    except ValueError:
                        score = "1.0"
                fout.write(f"{src}\t{tgt}\t{score}\n")

        # process remaining
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                src = parts[0].strip()
                tgt = parts[1].strip()
                score = "1.0"
                if len(parts) >= 4:
                    try:
                        score = str(float(parts[3].strip()))
                    except ValueError:
                        score = "1.0"
                fout.write(f"{src}\t{tgt}\t{score}\n")


def _ensure_deeponto_format(filepath: Path, tmpdir: str) -> Path:
    """return a DeepOnto-compatible TSV path, converting if necessary"""
    fmt = _detect_tsv_format(filepath)
    if fmt == "deeponto":
        return filepath
    tmp_path = Path(tmpdir) / f"converted_{filepath.name}"
    _convert_to_deeponto_tsv(filepath, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Custom data loading (format-agnostic, used for oracle metrics and fallback)
# ---------------------------------------------------------------------------

def load_mapping_pairs(filepath: Path) -> set[tuple[str, str]]:
    """
    load a mapping file as (source_uri, target_uri) pairs
    handles both DeepOnto format (header) and OAEI format (no header)
    """
    pairs = set()
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

        # skip header if present
        if not (first_line.startswith("SrcEntity") or first_line.startswith("src")):
            parts = first_line.split("\t")
            if len(parts) >= 2:
                pairs.add((parts[0].strip(), parts[1].strip()))

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.add((parts[0].strip(), parts[1].strip()))

    return pairs


def load_oracle_predictions(filepath: Path) -> list[dict]:
    """load oracle predictions CSV"""
    predictions = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_raw = row["Oracle_prediction"].strip()
            if pred_raw.lower() in ("true", "yes"):
                pred = True
            elif pred_raw.lower() in ("false", "no"):
                pred = False
            else:
                pred = None

            predictions.append({
                "source": row["source_entity_uri"].strip(),
                "target": row["target_entity_uri"].strip(),
                "prediction": pred,
                "confidence": float(row.get("Oracle_confidence", 0.0)),
            })
    return predictions


# ------------------------
# Global matching metrics
# ------------------------

def compute_global_deeponto(
    system_path: Path,
    reference_path: Path,
    train_reference_path: Path | None = None,
) -> dict:
    """compute global P, R, F1 using DeepOnto's AlignmentEvaluator"""
    EntityMapping = _deeponto_modules["EntityMapping"]
    ReferenceMapping = _deeponto_modules["ReferenceMapping"]
    AlignmentEvaluator = _deeponto_modules["AlignmentEvaluator"]

    with tempfile.TemporaryDirectory() as tmpdir:
        sys_converted = _ensure_deeponto_format(system_path, tmpdir)
        ref_converted = _ensure_deeponto_format(reference_path, tmpdir)

        preds = EntityMapping.read_table_mappings(str(sys_converted))
        refs = ReferenceMapping.read_table_mappings(str(ref_converted))

        kwargs = {}
        if train_reference_path and train_reference_path.exists():
            train_converted = _ensure_deeponto_format(train_reference_path, tmpdir)
            null_refs = ReferenceMapping.read_table_mappings(str(train_converted))
            kwargs["null_reference_mappings"] = null_refs

        results = AlignmentEvaluator.f1(preds, refs, **kwargs)

    # DeepOnto only returns P/R/F1; compute counts ourselves so the output 
    # is always complete regardless of eval backend
    system = load_mapping_pairs(system_path)
    reference = load_mapping_pairs(reference_path)
    if train_reference_path and train_reference_path.exists():
        train_pairs = load_mapping_pairs(train_reference_path)
        system = system - train_pairs
        reference = reference - train_pairs

    tp = len(system & reference)
    fp = len(system - reference)
    fn = len(reference - system)

    return {
        "precision": results.get("P", results.get("Precision", 0.0)),
        "recall": results.get("R", results.get("Recall", 0.0)),
        "f1": results.get("F1", results.get("F-score", 0.0)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "system_size": len(system),
        "reference_size": len(reference),
        "source": "deeponto",
    }




def compute_global_custom(
    system_path: Path,
    reference_path: Path,
    train_reference_path: Path | None = None,
) -> dict:
    """compute global P, R, F1 using custom implementation (fallback)"""
    system = load_mapping_pairs(system_path)
    reference = load_mapping_pairs(reference_path)

    if train_reference_path and train_reference_path.exists():
        train_pairs = load_mapping_pairs(train_reference_path)
        system = system - train_pairs
        reference = reference - train_pairs

    tp = len(system & reference)
    fp = len(system - reference)
    fn = len(reference - system)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "system_size": len(system),
        "reference_size": len(reference),
        "source": "custom",
    }


def compute_global_metrics(
    system_path: Path,
    reference_path: Path,
    train_reference_path: Path | None = None,
    force_custom: bool = False,
) -> dict:
    """compute global metrics, preferring DeepOnto when available"""
    if deeponto_available() and not force_custom:
        try:
            return compute_global_deeponto(system_path, reference_path, train_reference_path)
        except Exception as e:
            print(f"  DeepOnto evaluation failed ({e}), falling back to custom.", file=sys.stderr)

    return compute_global_custom(system_path, reference_path, train_reference_path)


# --------------------------------------
# Stratified global metrics (KG track)
# --------------------------------------

def _classify_uri_entity_type(uri: str) -> str:
    """classify a URI as class, property, or instance based on DBkWik patterns"""
    uri_lower = uri.lower()
    if '/class/' in uri_lower:
        return 'class'
    if '/property/' in uri_lower:
        return 'property'
    if '/resource/' in uri_lower:
        return 'instance'
    return 'unknown'


def compute_stratified_global_metrics(
    system_path: Path,
    reference_path: Path,
    stratified_refs: dict[str, Path] | None = None,
    partial_eval: bool = False,
) -> dict:
    """
    compute P/R/F1 stratified by entity type for KG track evaluation

    There are two strategies:

    1. If stratified_refs are provided (pre-split reference files per type),
       compute metrics by evaluating system output against each type's
       reference separately.

    2. Otherwise, classify URIs using DBkWik patterns and partition
       both system and reference for per-type evaluation.

    Parameters
    ----------
    system_path : Path
        Path to the system alignment TSV.
    reference_path : Path
        Path to the overall reference alignment TSV.
    stratified_refs : dict, optional
        Mapping of entity type names to reference TSV paths, e.g.:
        {
            "class": Path("reference_class.tsv"),
            "property": Path("reference_property.tsv"),
            "instance": Path("reference_instance.tsv")
        }
    partial_eval : bool
        If True, use PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE semantics
        (OAEI KG track evaluation protocol) instead of complete gold standard.

    Returns
    -------
    dict with keys 'overall', 'class', 'property', 'instance', each
    containing P/R/F1/TP/FP/FN metrics.
    """
    system_pairs = load_mapping_pairs(system_path)
    reference_pairs = load_mapping_pairs(reference_path)

    _eval_fn = compute_kg_partial_prf if partial_eval else _compute_prf

    results = {}

    # overall (unpartitioned)
    overall = _eval_fn(system_pairs, reference_pairs)
    if not partial_eval:
        overall["source"] = "custom_stratified"
    results["overall"] = overall

    if stratified_refs:
        # strategy 1: use pre-split reference files
        # classify ALL system pairs by URI pattern once
        sys_by_type = {"class": set(), "property": set(), "instance": set()}
        for pair in system_pairs:
            etype = _classify_uri_entity_type(pair[0])
            if etype in sys_by_type:
                sys_by_type[etype].add(pair)

        for entity_type, ref_path in stratified_refs.items():
            if ref_path.exists():
                type_ref = load_mapping_pairs(ref_path)
                type_sys = sys_by_type.get(entity_type, set())
                metrics = _eval_fn(type_sys, type_ref)
                if not partial_eval:
                    metrics["source"] = "custom_stratified"
                results[entity_type] = metrics
    else:
        # strategy 2: classify URIs by pattern
        type_buckets = {"class": (set(), set()),
                        "property": (set(), set()),
                        "instance": (set(), set())}

        for src, tgt in system_pairs:
            etype = _classify_uri_entity_type(src)
            if etype in type_buckets:
                type_buckets[etype][0].add((src, tgt))

        for src, tgt in reference_pairs:
            etype = _classify_uri_entity_type(src)
            if etype in type_buckets:
                type_buckets[etype][1].add((src, tgt))

        for entity_type, (sys_set, ref_set) in type_buckets.items():
            if ref_set:
                metrics = _eval_fn(sys_set, ref_set)
                if not partial_eval:
                    metrics["source"] = "custom_stratified"
                results[entity_type] = metrics

    return results


def _compute_prf(system: set, reference: set) -> dict:
    """compute P/R/F1 from set intersection"""
    tp = len(system & reference)
    fp = len(system - reference)
    fn = len(reference - system)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "system_size": len(system),
        "reference_size": len(reference),
    }


def compute_kg_partial_prf(
    system: set[tuple[str, str]],
    reference: set[tuple[str, str]],
) -> dict:
    """
    compute P/R/F1 under PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE semantics

    This implements the OAEI KG track evaluation protocol used by the MELT
    framework's ConfusionMatrixMetric.computeForPartialGoldStandard()

    The partial gold standard consists of 1:1 mappings extracted from cross-wiki links.
    
    For each reference pair <src, tgt>, we assume:
      - Source-complete: src's correct target is fully specified by the reference.
        Any system pair <src, X> where X != tgt is a false positive.
      - Target-complete: tgt's correct source is fully specified by the reference.
        Any system pair <Y, tgt> where Y != src is a false positive.

    System pairs where NEITHER entity appears in the reference are silently
    ignored (not counted as TP, FP, or FN), since the reference says nothing
    about those entities.

    Parameters
    ----------
    system : set of (src_uri, tgt_uri) tuples
    reference : set of (src_uri, tgt_uri) tuples

    Returns
    -------
    dict with precision, recall, f1, counts, and diagnostic fields
    """
    if not reference:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "true_positives": 0, "false_positives": 0, "false_negatives": 0,
            "ignored": len(system), "system_size": len(system),
            "evaluated_size": 0, "reference_size": 0,
            "source": "kg_partial",
        }

    # build source/target indices for both system and reference
    from collections import defaultdict

    ref_sources = defaultdict(set)   # src -> set of (src, tgt) pairs
    ref_targets = defaultdict(set)   # tgt -> set of (src, tgt) pairs
    for src, tgt in reference:
        ref_sources[src].add((src, tgt))
        ref_targets[tgt].add((src, tgt))

    sys_by_source = defaultdict(set)
    sys_by_target = defaultdict(set)
    for src, tgt in system:
        sys_by_source[src].add((src, tgt))
        sys_by_target[tgt].add((src, tgt))

    tp = 0
    fp_set = set()

    for ref_src, ref_tgt in reference:
        ref_pair = (ref_src, ref_tgt)

        # check if system contains this reference pair
        if ref_pair in system:
            tp += 1

        # source-complete: any system pair <ref_src, X> where X != ref_tgt is FP
        for sys_pair in sys_by_source.get(ref_src, set()):
            if sys_pair != ref_pair:
                fp_set.add(sys_pair)

        # target-complete: any system pair <Y, ref_tgt> where Y != ref_src is FP
        for sys_pair in sys_by_target.get(ref_tgt, set()):
            if sys_pair != ref_pair:
                fp_set.add(sys_pair)

    # remove TPs from FP set 
    # this handles multi-valued references where a pair (A,B) appears in refset 
    # and could be flagged FP via a different ref entry sharing A or B
    fp_set -= (system & reference)

    fp = len(fp_set)
    fn = len(reference) - tp

    # count ignored system pairs: neither src nor tgt appears in any reference
    all_ref_sources = set(ref_sources.keys())
    all_ref_targets = set(ref_targets.keys())
    ignored = sum(
        1 for s, t in system
        if s not in all_ref_sources and t not in all_ref_targets
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "ignored": ignored,
        "system_size": len(system),
        "evaluated_size": tp + fp,
        "reference_size": len(reference),
        "source": "kg_partial",
    }


# ------------------------------------------------
# Conference track M1/M2/M3 stratified evaluation
# ------------------------------------------------

def _load_entity_types_from_initial_alignment(initial_alignment_path: Path) -> dict:
    """
    load URI -> entity type mapping from LogMap's initial alignment file

    The initial alignment is a pipe-delimited file with columns:
        
        source_entity_uri | target_entity_uri | relation | confidence | entityType

    Where entityType is one of: CLS, OPROP, DPROP, INST, UNKNO.

    Returns a dict mapping URI -> entity type string.
    """
    uri_types = {}
    with open(initial_alignment_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 5:
                src_uri = parts[0].strip()
                tgt_uri = parts[1].strip()
                etype = parts[4].strip()
                uri_types[src_uri] = etype
                uri_types[tgt_uri] = etype
    return uri_types


def _classify_conference_pair(src_uri, tgt_uri, uri_types):
    """
    classify a mapping pair as 'class', 'property', or 'other'

    Uses the entityType information from the initial alignment.
    A mapping is 'class' if both URIs are CLS, 'property' if both are OPROP or DPROP, and 'other' otherwise.
    """
    src_type = uri_types.get(src_uri, "UNKNO")
    tgt_type = uri_types.get(tgt_uri, "UNKNO")

    if src_type == "CLS" and tgt_type == "CLS":
        return "class"
    if src_type in ("OPROP", "DPROP") and tgt_type in ("OPROP", "DPROP"):
        return "property"
    return "other"


def compute_conference_stratified_metrics(
    system_path: Path,
    reference_path: Path,
    initial_alignment_path: Path | None = None,
) -> dict:
    """
    compute M1 (class), M2 (property), M3 (all) metrics for Conference track

    Strategy:

    1. Look for pre-split reference files ({pair}_class.tsv, {pair}_property.tsv)
       alongside the main reference.  These are generated by prepare_conference_stratified_refs.py.

    2. Classify system output URIs using the initial alignment's entityType column if available, 
       otherwise fall back to reference-only evaluation.

    Parameters
    ----------
    system_path : Path
        System alignment TSV.
    reference_path : Path
        Full (M3) reference alignment TSV.
    initial_alignment_path : Path, optional
        LogMap initial alignment file (pipe-delimited) for entity type info.

    Returns
    -------
    dict with keys 'm1_class', 'm2_property', each containing P/R/F1/counts.
    Returns empty sub-dicts for variants where data is unavailable.
    """
    ref_dir = reference_path.parent
    ref_stem = reference_path.stem  # e.g. 'cmt-conference'

    class_ref_path = ref_dir / f"{ref_stem}_class.tsv"
    prop_ref_path = ref_dir / f"{ref_stem}_property.tsv"

    results = {}

    if not class_ref_path.exists() and not prop_ref_path.exists():
        return results

    system_pairs = load_mapping_pairs(system_path)

    # load entity type info for classifying system output
    uri_types = {}
    if initial_alignment_path and initial_alignment_path.exists():
        uri_types = _load_entity_types_from_initial_alignment(initial_alignment_path)

    # M1: class correspondences
    if class_ref_path.exists():
        class_ref = load_mapping_pairs(class_ref_path)

        if uri_types:
            # classify system pairs using entity type info
            class_system = {
                (s, t) for s, t in system_pairs
                if _classify_conference_pair(s, t, uri_types) == "class"
            }
        else:
            # fallback: only evaluate against class reference (no FP from system pairs we can't classify)
            class_system = system_pairs

        m1 = _compute_prf(class_system, class_ref)
        m1["source"] = "conference_stratified"
        results["m1_class"] = m1

    # M2: property correspondences
    if prop_ref_path.exists():
        prop_ref = load_mapping_pairs(prop_ref_path)

        if uri_types:
            prop_system = {
                (s, t) for s, t in system_pairs
                if _classify_conference_pair(s, t, uri_types) == "property"
            }
        else:
            prop_system = system_pairs

        m2 = _compute_prf(prop_system, prop_ref)
        m2["source"] = "conference_stratified"
        results["m2_property"] = m2

    return results



# ---------------------
# Local ranking metrics
# ---------------------

def compute_ranking_metrics(
    test_cands_path: Path | None = None,
    oracle_predictions_path: Path | None = None,
    Ks: list[int] | None = None,
) -> dict:
    """
    compute MRR and Hits@K using DeepOnto's ranking_eval.

    NOTE: Ranking evaluation requires the system to have scored ALL candidates in test.cands.tsv, 
    not just the m_ask subset. Currently, LogMap-LLM only scores the m_ask candidates, so this fn
    serves as a placeholder.

    When the full scoring pipeline is implemented (Task 1.5 integration), this will convert oracle 
    predictions to scored candidates format and call DeepOnto's ranking_eval.
    """
    if not deeponto_available():
        return {"error": "DeepOnto not installed — ranking metrics unavailable."}

    if Ks is None:
        Ks = [1, 5, 10]

    if test_cands_path is None or not test_cands_path.exists():
        return {"error": f"test.cands.tsv not found at {test_cands_path}"}

    # TODO: Convert oracle predictions -> scored candidates format
    # LogMap-LLM currently scores only m_ask (~259 candidates for anatomy),
    # not the full ~100 candidates per reference mapping needed here
    #
    # i.e., when ready:
    #
    #   results = ranking_eval(str(scored_path), has_score=True, Ks=Ks)
    #   return {"mrr": results["MRR"],
    #           **{f"hits@{k}": results[f"Hits@{k}"] for k in Ks}, "source": "deeponto"}

    return {"error": "Ranking evaluation requires full candidate scoring (not yet implemented)."}


# ------------------------------
# Oracle discrimination metrics
# ------------------------------

def compute_oracle_metrics(
    predictions: list[dict],
    reference: set[tuple[str, str]],
    is_kg_task: bool = False,
) -> dict:
    """
    compute oracle discrimination metrics on m_ask

    Confusion matrix:
      TP: oracle said True, mapping is in reference
      FP: oracle said True, mapping NOT in reference
      TN: oracle said False, mapping NOT in reference
      FN: oracle said False, mapping IS in reference

    When is_kg_task is True, uses three-way labelling under partial gold standard semantics:
      
      - POSITIVE: pair is in reference
      - NEGATIVE: pair not in reference but at least one entity appears in the 
                  reference (so its correct match is known and this isn't it)
      - UNEVALUABLE: neither entity appears in any reference pair 
                     (excluded from confusion matrix entirely)
    """
    tp = fp = tn = fn = 0
    errors = 0
    excluded = 0
    false_mappings = []

    # build evaluability indices for partial gold standard
    if is_kg_task:
        ref_sources = {src for src, tgt in reference}
        ref_targets = {tgt for src, tgt in reference}

    for pred in predictions:
        if pred["prediction"] is None:
            errors += 1
            continue

        pair = (pred["source"], pred["target"])
        in_reference = pair in reference
        predicted_true = pred["prediction"]

        # under partial gold standard, check evaluability
        if is_kg_task and not in_reference:
            if pred["source"] not in ref_sources and pred["target"] not in ref_targets:
                # neither entity appears in reference — unevaluable
                excluded += 1
                continue

        if predicted_true and in_reference:
            tp += 1
        elif predicted_true and not in_reference:
            fp += 1
            false_mappings.append({
                "source_entity_uri": pred["source"],
                "target_entity_uri": pred["target"],
                "oracle_prediction": True,
                "oracle_confidence": pred["confidence"],
                "in_reference": False,
                "error_type": "FP",
            })
        elif not predicted_true and not in_reference:
            tn += 1
        elif not predicted_true and in_reference:
            fn += 1
            false_mappings.append({
                "source_entity_uri": pred["source"],
                "target_entity_uri": pred["target"],
                "oracle_prediction": False,
                "oracle_confidence": pred["confidence"],
                "in_reference": True,
                "error_type": "FN",
            })

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    youdens_j = sensitivity + specificity - 1.0

    oracle_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    oracle_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    oracle_f1 = (
        2 * oracle_precision * oracle_recall / (oracle_precision + oracle_recall)
        if (oracle_precision + oracle_recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "errors": errors,
        "oracle_excluded": excluded,
        "total_candidates": len(predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youdens_j": youdens_j,
        "oracle_precision": oracle_precision,
        "oracle_recall": oracle_recall,
        "oracle_f1": oracle_f1,
        "false_mappings": false_mappings,
    }


# --------
# Display Metrics
# --------

def print_global_metrics(metrics: dict) -> None:
    print("Global Alignment Metrics")
    source = metrics.get("source", "unknown")
    print(f"  (computed via: {source})")
    print()
    if "system_size" in metrics:
        print(f"  System mappings:  {metrics['system_size']}")
        print(f"  Reference size:   {metrics['reference_size']}")
        print(f"  True positives:   {metrics['true_positives']}")
        print(f"  False positives:  {metrics['false_positives']}")
        print(f"  False negatives:  {metrics['false_negatives']}")
        if "ignored" in metrics:
            print(f"  Ignored (unevaluable): {metrics['ignored']}")
            print(f"  Evaluated size:   {metrics['evaluated_size']}")
        print()
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")


def print_ranking_metrics(metrics: dict) -> None:
    print("Local Ranking Metrics")
    if "error" in metrics:
        print(f"  {metrics['error']}")
        return
    print()
    if "mrr" in metrics:
        print(f"  MRR:        {metrics['mrr']:.4f}")
    for key, val in metrics.items():
        if key.startswith("hits@"):
            print(f"  {key.upper()}:    {val:.4f}")


def print_oracle_metrics(metrics: dict) -> None:
    print("Oracle Discrimination Metrics (on m_ask)")
    print()
    print(f"  Total candidates:   {metrics['total_candidates']}")
    print(f"  TP (accept, correct):   {metrics['tp']}")
    print(f"  FP (accept, wrong):     {metrics['fp']}")
    print(f"  TN (reject, correct):   {metrics['tn']}")
    print(f"  FN (reject, wrong):     {metrics['fn']}")
    if metrics["errors"] > 0:
        print(f"  Errors (unparseable):   {metrics['errors']}")
    if metrics.get("oracle_excluded", 0) > 0:
        print(f"  Excluded (unevaluable): {metrics['oracle_excluded']}")
    print()
    print(f"  Sensitivity (TPR):  {metrics['sensitivity']:.4f}")
    print(f"  Specificity (TNR):  {metrics['specificity']:.4f}")
    print(f"  Youden's J:         {metrics['youdens_j']:.4f}")
    print()
    print(f"  Oracle Precision:   {metrics['oracle_precision']:.4f}")
    print(f"  Oracle Recall:      {metrics['oracle_recall']:.4f}")
    print(f"  Oracle F1:          {metrics['oracle_f1']:.4f}")


# ------------------------
# Path resolution helpers
# ------------------------

def find_system_mappings(results_dir: Path, task_name: str) -> Path | None:
    refined_dir = results_dir / "logmap-refined-alignment"
    if not refined_dir.is_dir():
        return None
    tsv = refined_dir / f"{task_name}-logmap_mappings.tsv"
    if tsv.exists():
        return tsv
    for f in refined_dir.glob("*mappings.tsv"):
        return f
    return None


def find_oracle_predictions(results_dir: Path, task_name: str) -> Path | None:
    output_dir = results_dir / "logmapllm-outputs"
    if not output_dir.is_dir():
        return None
    for f in output_dir.glob("*predictions*.csv"):
        return f
    return None


def find_reference_alignment(datasets_dir: Path, task_name: str) -> Path | None:
    candidates = []
    if task_name == "anatomy":
        candidates.extend([
            datasets_dir / "oracles" / "Pipeline-Code-and-Runs" / "data"
            / "anatomy" / "human-mouse" / "refs_equiv" / "full.tsv",
            datasets_dir / "anatomy" / "refs_equiv" / "full.tsv",
        ])
    elif task_name.startswith("conference-"):
        # conference track: task_name is 'conference-{src}-{tgt}'
        parts = task_name.split("-", 1)  # ['conference', '{src}-{tgt}']
        if len(parts) == 2:
            pair_name = parts[1]  # e.g. 'cmt-conference'
            candidates.extend([
                datasets_dir / "conference" / "refs_equiv" / f"{pair_name}.tsv",
                # also check for original RDF in case TSV hasn't been generated
                datasets_dir / "conference" / "reference" / f"{pair_name}.rdf",
            ])
    elif task_name.startswith("kg-"):
        # KG track: task_name is 'kg-{pair_name}'
        pair_name = task_name[3:]  # e.g. 'starwars-swg'
        candidates.extend([
            datasets_dir / "knowledgegraph" / pair_name / "refs_equiv" / "reference_all.tsv",
            datasets_dir / "knowledgegraph" / pair_name / "refs_equiv" / "reference.tsv",
        ])
    else:
        candidates.extend([
            datasets_dir / "bio-ml" / task_name / "refs_equiv" / "full.tsv",
            datasets_dir / "oracles" / "Pipeline-Code-and-Runs" / "data"
            / "bioml-2024" / task_name / "refs_equiv" / "full.tsv",
        ])
    for c in candidates:
        if c.exists():
            return c
    return None


# ----------------------------
# Main evaluation entry point
# ----------------------------

VALID_METRICS = {"global", "ranking", "oracle"}
DEFAULT_METRICS = ["global", "oracle"]


def evaluate_alignment(
    system_mappings_path: str | Path,
    reference_path: str | Path,
    oracle_predictions_path: str | Path | None = None,
    train_reference_path: str | Path | None = None,
    test_cands_path: str | Path | None = None,
    initial_alignment_path: str | Path | None = None,
    task_name: str = "",
    metrics: list[str] | None = None,
    output_json_path: str | Path | None = None,
    force_custom: bool = False,
    is_kg_task: bool = False,
) -> dict:
    """
    run evaluation and return results dict

    Parameters
    ----------
    system_mappings_path : path
        LogMap refined alignment TSV
    reference_path : path
        Reference alignment (full.tsv or test.tsv)
    oracle_predictions_path : path, optional
        Oracle predictions CSV (for oracle metrics)
    train_reference_path : path, optional
        Training mappings TSV (for semi-supervised null exclusion)
    test_cands_path : path, optional
        Candidate mappings TSV (for ranking metrics)
    initial_alignment_path : path, optional
        LogMap initial alignment file (pipe-delimited) — used for
        Conference track M1/M2 stratification via entityType column
    task_name : str
        Task name for display
    metrics : list of str
        Which metric groups to compute: 'global', 'ranking', 'oracle'
    output_json_path : path, optional
        If provided, write results to this JSON file
    force_custom : bool
        If True, use custom evaluation even when DeepOnto is available
    is_kg_task : bool
        If True, use PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE evaluation
        semantics (OAEI KG track protocol). Bypasses DeepOnto and uses
        partial gold standard logic for both global and oracle metrics.
    """
    if metrics is None:
        metrics = list(DEFAULT_METRICS)

    invalid = set(metrics) - VALID_METRICS
    if invalid:
        print(f"WARNING: Unknown metric groups ignored: {invalid}", file=sys.stderr)
        metrics = [m for m in metrics if m in VALID_METRICS]

    system_mappings_path = Path(system_mappings_path)
    reference_path = Path(reference_path)

    # RDF fallback: if the reference is in OAEI RDF/XML format, convert to TSV
    if reference_path.suffix.lower() == '.rdf':
        try:
            from convert_oaei_rdf_to_tsv import convert_rdf_alignment
            converted_tsv = reference_path.with_suffix('.tsv')
            if not converted_tsv.exists():
                n = convert_rdf_alignment(reference_path, converted_tsv)
                print(f"Converted RDF reference to TSV: {converted_tsv} ({n} mappings)")
            reference_path = converted_tsv
        except ImportError:
            print("WARNING: Reference is RDF but convert_oaei_rdf_to_tsv not available.",
                  file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Failed to convert RDF reference: {e}", file=sys.stderr)

    train_ref = Path(train_reference_path) if train_reference_path else None
    test_cands = Path(test_cands_path) if test_cands_path else None
    oracle_path = Path(oracle_predictions_path) if oracle_predictions_path else None
    initial_align = Path(initial_alignment_path) if initial_alignment_path else None

    results = {"task_name": task_name}

    if is_kg_task:
        print()
        print("NOTE: KG track partial gold standard evaluation active.")
        print("      Using PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE semantics.")
    elif not deeponto_available() and ("global" in metrics or "ranking" in metrics):
        if force_custom:
            print()
            print("NOTE: Using custom evaluation (--no-deeponto).")
        else:
            print()
            print("NOTE: DeepOnto not installed. Global metrics will use custom")
            print("      implementation. Ranking metrics are unavailable.")
            print("      Install with: pip install deeponto")

    # GLOBAL MATCHING:
    if "global" in metrics:
        print()
        print("- - - - - - - - - - - - - - - - - - - - - - - -")

        if is_kg_task:
            # KG track: use partial gold standard evaluation
            system_pairs = load_mapping_pairs(system_mappings_path)
            reference_pairs = load_mapping_pairs(reference_path)
            global_metrics = compute_kg_partial_prf(system_pairs, reference_pairs)
            # Print reference size diagnostic for version verification
            print(f"  (KG partial eval — reference has {len(reference_pairs)} pairs)")
        else:
            global_metrics = compute_global_metrics(
                system_mappings_path, reference_path, train_ref,
                force_custom=force_custom,
            )

        results["global"] = global_metrics
        print_global_metrics(global_metrics)

        # --- STRATIFIED GLOBAL METRICS FOR KG TRACK ---
        if task_name.startswith("kg-") or is_kg_task:
            ref_dir = reference_path.parent
            stratified_refs = {}
            for etype in ("class", "property", "instance"):
                type_ref = ref_dir / f"reference_{etype}.tsv"
                if type_ref.exists():
                    stratified_refs[etype] = type_ref

            eval_label = "partial" if is_kg_task else "complete"
            if stratified_refs:
                print()
                print("- - - - - - - - - - - - - - - - - - - - - - - -")
                print(f"Stratified evaluation (by entity type, {eval_label} GS):")
                strat_metrics = compute_stratified_global_metrics(
                    system_mappings_path, reference_path, stratified_refs,
                    partial_eval=is_kg_task,
                )
            else:
                # fall back to URI-based classification
                print()
                print("- - - - - - - - - - - - - - - - - - - - - - - -")
                print(f"Stratified evaluation (URI-pattern classification, {eval_label} GS):")
                strat_metrics = compute_stratified_global_metrics(
                    system_mappings_path, reference_path,
                    partial_eval=is_kg_task,
                )

            for etype in ("class", "property", "instance"):
                if etype in strat_metrics:
                    m = strat_metrics[etype]
                    extra = ""
                    if "ignored" in m:
                        extra = f", Ign={m['ignored']}"
                    print(f"  {etype:>10s}:  P={m['precision']:.4f}  "
                          f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
                          f"(TP={m['true_positives']}, FP={m['false_positives']}, "
                          f"FN={m['false_negatives']}{extra})")
                    results[f"global_{etype}"] = m

        # --- STRATIFIED GLOBAL METRICS FOR CONFERENCE TRACK (M1/M2) ---
        if task_name.startswith("conference-"):
            conf_strat = compute_conference_stratified_metrics(
                system_mappings_path, reference_path,
                initial_alignment_path=initial_align,
            )
            if conf_strat:
                print()
                print("- - - - - - - - - - - - - - - - - - - - - - - -")
                print("Conference M1/M2 stratified evaluation:")
                for variant_key in ("m1_class", "m2_property"):
                    if variant_key in conf_strat:
                        m = conf_strat[variant_key]
                        label = "M1 (class)" if "m1" in variant_key else "M2 (property)"
                        print(f"  {label:>15s}:  P={m['precision']:.4f}  "
                              f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
                              f"(TP={m['true_positives']}, FP={m['false_positives']}, "
                              f"FN={m['false_negatives']})")
                        results[f"global_{variant_key}"] = m

    # LOCAL RANKING:
    if "ranking" in metrics:
        print()
        print("- - - - - - - - - - - - - - - - - - - - - - - -")
        ranking_metrics = compute_ranking_metrics(test_cands, oracle_path)
        results["ranking"] = ranking_metrics
        print_ranking_metrics(ranking_metrics)

    # ORACLE DISCRIMINATION:
    if "oracle" in metrics:
        if oracle_path and oracle_path.exists():
            reference_pairs = load_mapping_pairs(reference_path)
            predictions = load_oracle_predictions(oracle_path)
            oracle_metrics = compute_oracle_metrics(
                predictions, reference_pairs, is_kg_task=is_kg_task,
            )
            results["oracle"] = oracle_metrics
            print()
            print("- - - - - - - - - - - - - - - - - - - - - - - -")
            print_oracle_metrics(oracle_metrics)

            # write false_mappings.csv
            false_mappings = oracle_metrics.get("false_mappings", [])
            if false_mappings:
                if output_json_path:
                    fm_path = Path(output_json_path).parent / "false_mappings.csv"
                else:
                    fm_path = Path(oracle_path).parent / "false_mappings.csv"
                fm_path.parent.mkdir(parents=True, exist_ok=True)
                _fieldnames = [
                    "source_entity_uri", "target_entity_uri",
                    "oracle_prediction", "oracle_confidence",
                    "in_reference", "error_type",
                ]
                with open(fm_path, "w", newline="") as fm_f:
                    writer = csv.DictWriter(fm_f, fieldnames=_fieldnames)
                    writer.writeheader()
                    writer.writerows(false_mappings)
                print()
                print(f"False mappings ({len(false_mappings)} FP+FN) saved to: {fm_path}")
        else:
            print()
            print("Oracle predictions not found — skipping oracle metrics.")

    # --- WRITE JSON
    if output_json_path:
        # exclude the false_mappings list from the JSON to avoid bloat
        # it is written separately as false_mappings.csv above
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict) and "false_mappings" in value:
                value = {k: v for k, v in value.items() if k != "false_mappings"}
            json_results[key] = value
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print()
        print(f"Evaluation results saved to: {output_json_path}")

    return results


# ----
# CLI
# ----

def main():
    parser = argparse.ArgumentParser(description="Evaluate LogMap-LLM alignment results")

    parser.add_argument("--results-dir", type=Path, default=None,
        help="Results directory (auto-discovers system mappings and oracle predictions)")
    
    parser.add_argument("--datasets-dir", type=Path, default=None,
        help="Datasets directory (for auto-discovering reference alignment)")
    
    parser.add_argument("--system-mappings", type=Path, default=None,
        help="Path to refined alignment TSV (overrides --results-dir)")
    
    parser.add_argument("--oracle-predictions", type=Path, default=None,
        help="Path to oracle predictions CSV (overrides --results-dir)")
    
    parser.add_argument("--reference", type=Path, default=None,
        help="Path to reference alignment")
    
    parser.add_argument("--train-reference", type=Path, default=None,
        help="Path to training mappings (for semi-supervised null exclusion)")
    
    parser.add_argument("--test-cands", type=Path, default=None,
        help="Path to test candidate mappings (for ranking evaluation)")
    
    parser.add_argument("--initial-alignment", type=Path, default=None,
        help="Path to LogMap initial alignment (pipe-delimited) for "
             "Conference M1/M2 stratification")

    parser.add_argument("--task-name", "-t", type=str, required=True,
        help="Task name (e.g. anatomy, snomed-fma.body, conference-cmt-ekaw)")
    
    parser.add_argument("--track", type=str, default=None,
        choices=["conference", "bioml", "anatomy", "knowledgegraph"],
        help="Track hint for auto-discovery (optional)")
    
    parser.add_argument("--metrics", "-m", nargs="+",
        choices=sorted(VALID_METRICS), default=list(DEFAULT_METRICS),
        help="Metric groups to compute (default: global oracle)")
    
    parser.add_argument("--output-json", "-o", type=Path, default=None,
        help="Write evaluation results to JSON file")
    
    parser.add_argument("--no-deeponto", action="store_true", default=False,
        help="Use custom evaluation even if DeepOnto is installed")
    
    parser.add_argument("--kg-task", action="store_true", default=False,
        help="Use partial gold standard evaluation (OAEI KG track semantics: "
             "PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE)")
    
    parser.add_argument("--jvm-memory", type=str, default=None,
        help="JVM memory for DeepOnto (default: 8g, or JAVA_MEMORY env var)")

    args = parser.parse_args()

    # END: ARGPARSE

    # override JVM memory if provideded as CLI argument
    if args.jvm_memory:
        for var_name in ("JAVA_MEMORY", "DEEPONTO_JVM_MEMORY", "JVM_MEMORY"):
            os.environ[var_name] = args.jvm_memory

    # apply --no-deeponto flag via module-level toggle
    global _DEEPONTO_DISABLED
    if args.no_deeponto:
        _DEEPONTO_DISABLED = True

    system_path = args.system_mappings
    if system_path is None and args.results_dir:
        system_path = find_system_mappings(args.results_dir, args.task_name)
    if system_path is None or not system_path.exists():
        print("ERROR: Could not find system mappings.", file=sys.stderr)
        sys.exit(1)

    ref_path = args.reference
    if ref_path is None and args.datasets_dir:
        ref_path = find_reference_alignment(args.datasets_dir, args.task_name)
    if ref_path is None or not ref_path.exists():
        print("ERROR: Could not find reference alignment.", file=sys.stderr)
        sys.exit(1)

    oracle_path = args.oracle_predictions
    if oracle_path is None and args.results_dir:
        oracle_path = find_oracle_predictions(args.results_dir, args.task_name)

    output_json = args.output_json
    if output_json is None and args.results_dir:
        output_json = args.results_dir / "evaluation_results.json"

    print(f"Evaluating: {args.task_name}")
    print(f"  System:    {system_path}")
    print(f"  Reference: {ref_path}")
    if oracle_path:
        print(f"  Oracle:    {oracle_path}")
    if args.train_reference:
        print(f"  Train ref: {args.train_reference}")
    if args.test_cands:
        print(f"  Cands:     {args.test_cands}")
    if args.initial_alignment:
        print(f"  Init align:{args.initial_alignment}")
    print(f"  Metrics:   {args.metrics}")

    # detect KG task: explicit --kg-task flag takes precedence,
    # then fall back to --track knowledgegraph for backwards compatibility
    is_kg = args.kg_task or (args.track and args.track.lower() == 'knowledgegraph')
    if is_kg:
        print(f"  KG partial eval: active")

    evaluate_alignment(
        system_mappings_path=system_path,
        reference_path=ref_path,
        oracle_predictions_path=oracle_path,
        train_reference_path=args.train_reference,
        test_cands_path=args.test_cands,
        initial_alignment_path=args.initial_alignment,
        task_name=args.task_name,
        metrics=args.metrics,
        output_json_path=output_json,
        force_custom=args.no_deeponto,
        is_kg_task=is_kg,
    )


if __name__ == "__main__":
    main()