"""
logmap_llm.evaluation.io
file io operations for alignment evaluation
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from logmap_llm.constants import DEEPONTO_TSV_HEADER_PREFIXES
from logmap_llm.utils.data import normalise_prediction_column



def _is_header_line(first_field: str) -> bool:
    """
    check the first field in a TSV to see if it qualifies as a HEADER
    for an evaluation mappings file (ie. has been converted to DeepOnto
    TSV format for evaluation purposes - usually)
    """
    return first_field in DEEPONTO_TSV_HEADER_PREFIXES



def load_mapping_pairs(path: Path, sep: str = "\t") -> set[tuple[str, str]]:
    """
    loads an alignment file as a set of (source URI, target URI) pairs
    this supports three common shapes (auto-detected from the first line)

        1. DeepOnto-style TSV \w 'SecEntity\tTgtEntity\tScore'
        2. OAEI-style TSV without a header row
        3. LogMap output, also no header

    Note: this function does not consider additional columns, such as
    confidence, relation type, entity type, etc. it just loads the mappings
    """
    mapping_pairs_from_disk: set[tuple[str, str]] = set()
    with open(path) as fp:
        first_line = fp.readline().rstrip("\n")
        first_line_fields = first_line.split(sep)

        if first_line_fields and not _is_header_line(first_line_fields[0].strip()):
            # the first line is data, extract it, then loop
            if len(first_line_fields) >= 2:
                mapping_pairs_from_disk.add(
                    (
                        first_line_fields[0].strip(), 
                        first_line_fields[1].strip()
                    )
                )
        # all other lines:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            # else:
            parts = line.split(sep)
            if len(parts) >= 2:
                mapping_pairs_from_disk.add(
                    (
                        parts[0].strip(),
                        parts[1].strip()
                    )
                )
    return mapping_pairs_from_disk



def _coerce_prediction(value) -> bool | None:
    """
    coerces an 'Oracle_prediction' 
    f : (True, False, 'error', 'skipped', NaN, None, *) -> bool | None
    """
    if value is True or value is False:
        return value
    return None


def _coerce_confidence(value) -> float:
    """
    coerces an 'Oracle_confidence' column to a finite float
    failure modes: None (field missing), NaN, non-numeric string
    """
    if value is None:
        return 0.0
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_oracle_predictions(filepath: Path) -> list[dict]:
    """
    loads oracle predictions CSV into a list of prediction dicts
    CSVs must have (or are expected to have) at least three columns:
    
        - source_entity_uri : str
        - target_entity_uri : str
        - Oracle_prediction : bool | Literal['error', 'skipped']

        - Oracle_confidence : float [0, 1] | NaN [OPTIONAL?]
    
    then additional columns, such as oracle_input_tokens, oracle_output_tokens,
    original m_ask fields; are preserved (in the CSV), but ignored 
    by this fn (they are used for downstream purposes -- analysis, etc.)

    the list[dict] returned is a list of prediction dicts, where each prediction 
    dict has shape:
        {
            "source":     str,           # source_entity_uri
            "target":     str,           # target_entity_uri
            "prediction": bool | None,   # None means "consultation failed"
            "confidence": float,         # always finite, 0.0 on any error
        }
    """
    df = pd.read_csv(filepath)
    df = normalise_prediction_column(df)

    predictions: list[dict] = []

    for _idx, row in df.iterrows():
        predictions.append({
            'source': str(row["source_entity_uri"]).strip(),
            'target': str(row["target_entity_uri"]).strip(),
            'prediction': _coerce_prediction(row["Oracle_prediction"]),
            'confidence': _coerce_confidence(row.get("Oracle_confidence")),
        })

    return predictions




def load_entity_types_from_initial_alignment(initial_alignment_path: Path) -> dict[str, str]:
    """
    given a LogMap initial alignment file \w the m_ask format, ie.
        source|target|relation|confidence|entityType
    produce a map : URI -> entity_type; used by the conference 
    stratified-metrics path to classify mapping pairs as:
         - M1 (class) 
         - M2 (property)
    returns an empty dict if the file does not exist or is not parseable
    """
    uri_types: dict[str, str] = {}
    path = Path(initial_alignment_path)
    if not path.exists():
        return uri_types

    with open(path) as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 5:
                entity_type = parts[4].strip()
                uri_types[parts[0].strip()] = entity_type
                uri_types[parts[1].strip()] = entity_type

    return uri_types



def detect_tsv_format(filepath: Path) -> str:
    """
    observe the first line of a TSV file and return a short format tag:
        - deeeponto (if the file starts \w deeponto header)
        - else: 'oaei'
    used to decide whether a TSV file must be 
    converted prior to using DeepOnto
    """
    with open(filepath) as fp:
        first_line = fp.readline().rstrip("\n")
    first_field = first_line.split("\t", 1)[0].strip()
    if _is_header_line(first_field):
        return "deeponto"
    # else:
    return "oaei"



def convert_to_deeponto_tsv(input_path: Path, output_path: Path) -> None:
    """
    rewrites an 'OAEI-style' TSV (\w no header, variable column count) 
    as a deeponto style TSV \w header + three columns
    the file @ input_path should have at least 2 tab sep'd fields per row
    representing (source, target) mappings; if a fourth column is present
    we assume it contains a confidence score and is parsed as such,
    otherwise score defaults to 1.0; remaining fields are disgarded
    writes a new file at 'output_path' \w void return
    """
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        f_out.write("SrcEntity\tTgtEntity\tScore\n")
        for raw_line in f_in:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            src = parts[0].strip()
            tgt = parts[1].strip()
            score = parts[3].strip() if len(parts) > 3 else "1.0"
            try:
                float(score)
            except ValueError:
                score = "1.0"
            f_out.write(f"{src}\t{tgt}\t{score}\n")



def ensure_deeponto_format(filepath: Path, tmpdir: str) -> Path:
    """
    returns a filepath to a deeponto TSV version of the input file
    @ the filepath argument (assuming it matches the neccesary format)
    the neccesary format is described in comments in previous 
    fn doc-strings (above)
    """
    if detect_tsv_format(filepath) == "deeponto":
        return filepath
    converted = Path(tmpdir) / f"converted_{filepath.name}"
    convert_to_deeponto_tsv(filepath, converted)
    return converted

