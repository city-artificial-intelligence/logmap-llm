"""
logmap_llm.ontology.cache — owlready2 quadstore caching interface

Caches the parsed RDF quadstore to a SQLite file on disk so that repeated 
pipeline runs against the same ontology skip another OWL/RDF parse.  

Uses a build-once gate with copy-on-open strategy to avoid SQLite write-lock
contention between concurrent processes.

Canonical cache location:  ~/.cache/logmap-llm/owlready2/
Process-private copies:    /tmp/logmap-llm-owlcache-*/

To clear cached versions, see `clear_owlready2_cache.py` under `/scripts/`.
"""
from __future__ import annotations

import atexit
import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import owlready2

from logmap_llm.log_utils import info, step, success


# owlready2 World persistence cache
# ---------------------------------
# caches the parsed RDF quadstore to a SQLite file on disk so that repeated pipeline runs 
# against the same ontology skip the expensive OWL/RDF parse; uses a build-once gate with 
# copy-on-open strategy to avoid SQLite write-lock contention between concurrent processes
# ---
# default cache location: ~/.cache/logmap-llm/owlready2/
# process-private copies: /tmp/logmap-llm-owlcache-*/
# ---

# NOTE: we're probably using a more complex solution than is neccesarily required here.
# while this solution (owlready2 + a cache with readwrite locking) was originally optimal
# before making changes to LogMap for property detection, it should be possible to expose
# Java methods that we can use instead of this solution. TODO: this should be revisited.

_TEMP_DIR_PREFIX = 'logmap-llm-owlcache-'


# global registry of tmp directories this process creates for cleanup
# TODO: look into whether this is a potential cause of disk util
_temp_dirs_to_cleanup: list[str] = []


def _register_temp_cleanup(temp_dir: str) -> None:
    _temp_dirs_to_cleanup.append(temp_dir)


def _cleanup_temp_dirs() -> None:
    for d in _temp_dirs_to_cleanup:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def _sanitise_filename(name: str, max_len: int = 80) -> str:
    sanitised = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return sanitised[:max_len]


def _canonical_cache_path(onto_filepath: str, cache_dir: str) -> Path:
    """
    obtain a deterministic cache path for an ontology file

    returns a Path like:
        <cache_dir>/<sanitised_name>_<short_hash>.sqlite3

    short hash ensures collision safety when different directories contain 
    identically-named ontology files

    TODO: make configurable cache paths via config
    """
    resolved = str(Path(onto_filepath).resolve())
    short_hash = hashlib.sha256(resolved.encode()).hexdigest()[:12]
    base_name = Path(onto_filepath).name
    sanitised = _sanitise_filename(base_name)
    cache_filename = f'{sanitised}_{short_hash}.sqlite3'
    return Path(cache_dir) / cache_filename


def _is_cache_valid(onto_filepath: str, cache_path: Path) -> bool:
    """
    check whether a canonical cache file is valid (exists, non-empty, newer than src onto file)
    """
    if not cache_path.exists():
        return False
    if cache_path.stat().st_size == 0:
        return False
    source_mtime = Path(onto_filepath).stat().st_mtime
    cache_mtime = cache_path.stat().st_mtime
    return cache_mtime >= source_mtime


# TODO: move to appropriate helper location (this is duplicated elsewhere)
def _format_age(seconds: float) -> str:
    """format a duration in seconds as a human-readable age string"""
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.0f}m'
    # else:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f'{hours}h {minutes}m'


def _build_or_wait_for_cache(onto_filepath: str, cache_dir: str) -> Path:
    """
    build-once gate 
    ensures a valid 'default' cache exists

    uses an exclusive file lock so that exactly one process builds the cache
    other concurrent processes block until it is ready, then verify the result and proceed
    returns the Path to the default cache file
    """
    cache_path = _canonical_cache_path(onto_filepath, cache_dir)
    lock_path = cache_path.with_suffix('.sqlite3.lock')
    os.makedirs(cache_dir, exist_ok=True)

    onto_display = Path(onto_filepath).name

    # cache already valid, no lock needed (happy path)
    if _is_cache_valid(onto_filepath, cache_path):
        age = time.time() - cache_path.stat().st_mtime
        success(f'Using cached quadstore for {onto_display} (cache age: {_format_age(age)})')
        return cache_path

    # need to acquire lock and potentially build
    lock_fd = open(lock_path, 'w')
    try:
        info(f'Acquiring cache lock for {onto_display} ...')
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # check after acquiring the lock (another process may have built the cache while we were waiting)
        if _is_cache_valid(onto_filepath, cache_path):
            age = time.time() - cache_path.stat().st_mtime
            success(f'Using cached quadstore for {onto_display} (built by another process, age: {_format_age(age)})')
            return cache_path

        # (we're building) parse into a temp file in the same directory 
        # NOTE: we use the same filesystem to atomically rename
        step(f'Building owlready2 cache for {onto_display} ...')
        build_start = time.time()

        temp_fd, temp_build_path = tempfile.mkstemp(suffix='.sqlite3.tmp', dir=cache_dir)
        os.close(temp_fd)  # owlready2 opens the file itself

        try:
            build_world = owlready2.World(filename=temp_build_path)
            build_world.get_ontology(str(onto_filepath)).load()
            build_world.save()
            
            # close the world's SQLite connection before renaming
            build_world.close()

            # atomic rename (either completes or fails)
            os.replace(temp_build_path, str(cache_path))

            elapsed = time.time() - build_start
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            success(f'Cache built for {onto_display} in {elapsed:.1f}s ({size_mb:.0f} MB)')
        
        except Exception:
            # cleanup on failure
            with contextlib.suppress(OSError):
                os.unlink(temp_build_path)
            raise

        return cache_path

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _copy_to_private_temp(cache_path: Path) -> str:
    """
    copy a default cache file to a process-private temp directory
    returns the path to the priv copy. registers tmp dir for cleanup
    """
    temp_dir = tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX)
    _register_temp_cleanup(temp_dir)
    private_path = os.path.join(temp_dir, cache_path.name)
    shutil.copy2(str(cache_path), private_path)
    info(f'Working copy: {private_path}')
    return private_path


def _get_cached_world(onto_filepath: str, cache_dir: str) -> owlready2.World:
    """
    obtain an owlready2 world backed by a cached quadstore, handles:
    1. build-or-wait for the default cache
    2. copy to a private tmp file
    3. open private copy
    """
    canonical = _build_or_wait_for_cache(onto_filepath, cache_dir)
    private_path = _copy_to_private_temp(canonical)
    world = owlready2.World(filename=private_path)
    return world




# prefix used for process-private temp directories under /tmp
_TEMP_DIR_PREFIX = 'logmap-llm-owlcache-'

# global registry of temp directories created by this process
_temp_dirs_to_cleanup: list[str] = []

# default canonical cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
    'logmap-llm', 'owlready2',
)


def _register_temp_cleanup(temp_dir: str) -> None:
    _temp_dirs_to_cleanup.append(temp_dir)


def _cleanup_temp_dirs() -> None:
    for d in _temp_dirs_to_cleanup:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_temp_dirs)


def _sanitise_filename(name: str, max_len: int = 80) -> str:
    sanitised = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return sanitised[:max_len]


def _canonical_cache_path(onto_filepath: str, cache_dir: str) -> Path:
    resolved = str(Path(onto_filepath).resolve())
    short_hash = hashlib.sha256(resolved.encode()).hexdigest()[:12]
    base_name = Path(onto_filepath).name
    sanitised = _sanitise_filename(base_name)
    cache_filename = f'{sanitised}_{short_hash}.sqlite3'
    return Path(cache_dir) / cache_filename


def _is_cache_valid(onto_filepath: str, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    if cache_path.stat().st_size == 0:
        return False
    source_mtime = Path(onto_filepath).stat().st_mtime
    cache_mtime = cache_path.stat().st_mtime
    return cache_mtime >= source_mtime


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.0f}m'
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f'{hours}h {minutes}m'


def _build_or_wait_for_cache(onto_filepath: str, cache_dir: str) -> Path:
    cache_path = _canonical_cache_path(onto_filepath, cache_dir)
    lock_path = cache_path.with_suffix('.sqlite3.lock')
    os.makedirs(cache_dir, exist_ok=True)
    onto_display = Path(onto_filepath).name

    if _is_cache_valid(onto_filepath, cache_path):
        age = time.time() - cache_path.stat().st_mtime
        success(f'Using cached quadstore for {onto_display} (cache age: {_format_age(age)})')
        return cache_path

    lock_fd = open(lock_path, 'w')
    try:
        info(f'Acquiring cache lock for {onto_display} ...')
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        if _is_cache_valid(onto_filepath, cache_path):
            age = time.time() - cache_path.stat().st_mtime
            success(f'Using cached quadstore for {onto_display} '
                    f'(built by another process, age: {_format_age(age)})')
            return cache_path

        step(f'Building owlready2 cache for {onto_display} ...')
        build_start = time.time()

        temp_fd, temp_build_path = tempfile.mkstemp(suffix='.sqlite3.tmp', dir=cache_dir)
        os.close(temp_fd)

        try:
            build_world = owlready2.World(filename=temp_build_path)
            build_world.get_ontology(str(onto_filepath)).load()
            build_world.save()
            build_world.close()
            os.replace(temp_build_path, str(cache_path))

            elapsed = time.time() - build_start
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            success(f'Cache built for {onto_display} in {elapsed:.1f}s ({size_mb:.0f} MB)')
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(temp_build_path)
            raise

        return cache_path
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _copy_to_private_temp(cache_path: Path) -> str:
    temp_dir = tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX)
    _register_temp_cleanup(temp_dir)
    private_path = os.path.join(temp_dir, cache_path.name)
    shutil.copy2(str(cache_path), private_path)
    info(f'Working copy: {private_path}')
    return private_path


def _get_cached_world(onto_filepath: str, cache_dir: str) -> owlready2.World:
    canonical = _build_or_wait_for_cache(onto_filepath, cache_dir)
    private_path = _copy_to_private_temp(canonical)
    world = owlready2.World(filename=private_path)
    return world


class OntologyCache:
    """
    High-level cache interface injected into OntologyAccess
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = str(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    def get_cached_world(self, urionto: str):
        """return an owlready2.World backed by a cached quadstore"""
        return _get_cached_world(urionto, self.cache_dir)
