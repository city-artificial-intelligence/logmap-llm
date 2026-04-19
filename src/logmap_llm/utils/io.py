'''
IO utils
general-purpose IO module (at the project level)
'''

import json
from pathlib import Path
import hashlib

from logmap_llm.utils.logging import debug
from logmap_llm.constants import (
    DEFAULT_ENTROPY_CACHE_DIR,
    JSON_DATA,
    VERBOSE,
)



def atomic_json_write(path: Path, obj: JSON_DATA) -> None:
    '''
    Accepts a Path (expects `*/*.json`) and any valid python object that constitutes valid JSON data,
    tries to dump the file (with the suffix `.tmp`); if successful, will write to the specified 'path'
    (used in caching operations, eg. entropy caching).
    '''
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as tmp_write_loc:
            # fail loudly if NaN
            json.dump(obj, tmp_write_loc, ensure_ascii=False, sort_keys=True, allow_nan=False)
        tmp.replace(path)
    except OSError:
        pass



# TODO: consider migrating to 'more appropriate' location (?)
def compute_entropy_disk_cache_path(local_onto_fp: str, uri_pattern: str, entropy_cache_path: Path | str = DEFAULT_ENTROPY_CACHE_DIR) -> Path | None:
    '''
    Motivation: Experimental runs \w many large KGs take a long time to run
    we obtain very nice efficiency gains by writing the entropies to disk.
    However, we must ensure that they caching mechanism invalidates the cache
    if anything changes, therefore we use 
    
    Returns a content-addressed path for persisting entropies (or None).
    WARNING: will probably cause problems (actually, it _should_ skip) if 
    the ontology was not loaded from disk (eg. if resolving an ontology via
    HTTP \w a URL).

    Changes to the source file invalidates the cache. Cache validation/invalidation 
    hash calculated as SHA-256 (of the utf-8 encoded string):
    
        "'onto_path'|'onto_file_size'|'onto_file_modified'|'uri_matching_pattern'"

    TODO: migrate to a ProjectPaths obj (similar to PipelinePaths).
    '''
    entropy_cache_path = Path(entropy_cache_path)
    try:
        onto_path = Path(local_onto_fp).resolve()
        if not onto_path.is_file():
            return None
        # see: from io import stat, stat_result
        onto_stat = onto_path.stat()
    except (OSError, AttributeError):
        if VERBOSE:
            debug("(compute_entropy_disk_cache_path) Encountered OSError or AttributeError.")
        return None

    encoded_hash_input = (
        f"{onto_path}|{onto_stat.st_size}|{onto_stat.st_mtime_ns}|{uri_pattern}"
    ).encode()

    if VERBOSE:
        debug(f"(compute_entropy_disk_cache_path) ENCODED HASH INPUT: {encoded_hash_input}")

    hash_digest = hashlib.sha256(encoded_hash_input).hexdigest()[:16]

    full_cache_filepath = (entropy_cache_path / f"{onto_path.stem}_{hash_digest}.json").expanduser().resolve()

    if VERBOSE:
        debug(f"(compute_entropy_disk_cache_path) COMPUTED HASH DIGEST (OUTPUT): {hash_digest}")
        debug(f"(compute_entropy_disk_cache_path) entropy_cache_path: {full_cache_filepath}")

    return full_cache_filepath

