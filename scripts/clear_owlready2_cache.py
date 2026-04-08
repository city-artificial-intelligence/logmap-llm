import argparse
import glob
import os
import shutil
from pathlib import Path

###
# CANONICAL_DIR
# -------------
# The canonical location where the central owlready2 cache is stored
# (assumes a unix-based OS)
# TODO: We need to flip between unix and windows based locations
###

CANONICAL_DIR = os.path.expanduser('~/.cache/logmap-llm/owlready2')

###
# TMP LOCATION
# ------------
# When multiple processes run in parallel, each process should copy 
# its own version to /tmp/logmap-llm-owlcache-* to avoid readwrite 
# contention and to ensure each process has its own isolated copy
###

TMP_GLOB = '/tmp/logmap-llm-owlcache-*'


def _size_str(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if size_bytes < 1024:
        return f'{size_bytes} B'
    elif size_bytes < 1024 ** 2:
        return f'{size_bytes / 1024:.1f} KB'
    elif size_bytes < 1024 ** 3:
        return f'{size_bytes / (1024 ** 2):.1f} MB'
    else:
        return f'{size_bytes / (1024 ** 3):.2f} GB'


def _dir_size(path: str) -> int:
    """Recursively compute the total size of a directory."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def clear_canonical(dry_run: bool) -> int:
    """Remove canonical cache files.  Returns bytes reclaimed."""
    reclaimed = 0
    if not os.path.isdir(CANONICAL_DIR):
        print(f'  No canonical cache directory found at {CANONICAL_DIR}')
        return 0

    for entry in sorted(Path(CANONICAL_DIR).iterdir()):
        # removes .sqlite3 cache files and their .lock files
        if entry.suffix in ('.sqlite3', '.lock') or entry.name.endswith('.sqlite3.tmp'):
            size = entry.stat().st_size if entry.is_file() else 0
            if dry_run:
                print(f'  [dry-run] Would remove: {entry} ({_size_str(size)})')
            else:
                entry.unlink()
                print(f'  Removed: {entry} ({_size_str(size)})')
            reclaimed += size

    return reclaimed


def clear_tmp(dry_run: bool) -> int:
    """Remove stale process-private temp directories.  Returns bytes reclaimed."""
    reclaimed = 0
    tmp_dirs = sorted(glob.glob(TMP_GLOB))
    if not tmp_dirs:
        print(f'  No temp directories matching {TMP_GLOB}')
        return 0

    for d in tmp_dirs:
        size = _dir_size(d)
        if dry_run:
            print(f'  [dry-run] Would remove: {d}/ ({_size_str(size)})')
        else:
            shutil.rmtree(d, ignore_errors=True)
            print(f'  Removed: {d}/ ({_size_str(size)})')
        reclaimed += size

    return reclaimed


def main():
    parser = argparse.ArgumentParser(
        description='Clear owlready2 quadstore cache files.'
    )
    parser.add_argument(
        '--canonical-only',
        action='store_true',
        help='Only clear the shared cache under ~/.cache/logmap-llm/owlready2/'
    )
    parser.add_argument(
        '--tmp-only',
        action='store_true',
        help='Only clear stale process copies under /tmp/logmap-llm-owlcache-*/'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without deleting'
    )
    args = parser.parse_args()

    if args.canonical_only and args.tmp_only:
        parser.error('--canonical-only and --tmp-only are mutually exclusive')

    do_canonical = not args.tmp_only
    do_tmp = not args.canonical_only

    total_reclaimed = 0

    if do_canonical:
        print('Canonical cache:')
        total_reclaimed += clear_canonical(args.dry_run)

    if do_tmp:
        print('Temp directories:')
        total_reclaimed += clear_tmp(args.dry_run)

    action = 'Would reclaim' if args.dry_run else 'Total reclaimed'
    print(f'\n{action}: {_size_str(total_reclaimed)}')


if __name__ == '__main__':
    main()