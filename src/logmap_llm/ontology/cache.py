"""
logmap_llm.ontology.cache — owlready2 quadstore persistence cache.

Caches the parsed RDF quadstore to a SQLite file on disk so that
repeated pipeline runs against the same ontology skip the expensive
OWL/RDF parse.  Uses a build-once gate with copy-on-open strategy
to avoid SQLite write-lock contention between concurrent processes.

Canonical cache location:  ~/.cache/logmap-llm/owlready2/
Process-private copies:    /tmp/logmap-llm-owlcache-*/
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

from logmap_llm.utils.logging import info, success

# prefix used for process-private temp directories under /tmp
_TEMP_DIR_PREFIX = 'logmap-llm-owlcache-'

# global registry of temp directories created by this process
_temp_dirs_to_cleanup: list[str] = []

# default canonical cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get(
        'XDG_CACHE_HOME', 
        os.path.expanduser('~/.cache')
    ),
    'logmap-llm',
    'owlready2',
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
        info(f'Acquiring cache lock for {onto_display} ...', important=True)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        if _is_cache_valid(onto_filepath, cache_path):
            age = time.time() - cache_path.stat().st_mtime
            success(f'Using cached quadstore for {onto_display} '
                    f'(built by another process, age: {_format_age(age)})')
            return cache_path

        info(f'Building owlready2 cache for {onto_display} ...', important=True)
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
    info(f'Working copy: {private_path}', important=True)
    return private_path


def _get_cached_world(onto_filepath: str, cache_dir: str) -> owlready2.World:
    canonical = _build_or_wait_for_cache(onto_filepath, cache_dir)
    private_path = _copy_to_private_temp(canonical)
    world = owlready2.World(filename=private_path)
    return world


class OntologyCache:
    """High-level cache interface injected into OntologyAccess."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = str(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    def get_cached_world(self, urionto: str):
        """Return an owlready2.World backed by a cached quadstore."""
        return _get_cached_world(urionto, self.cache_dir)
