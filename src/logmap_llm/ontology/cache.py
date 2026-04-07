"""
logmap_llm.ontology.cache — owlready2 quadstore caching interface
NOTE: stub
"""
from __future__ import annotations

from pathlib import Path


class OntologyCache:

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def get_cached_world(self, urionto: str):
        return None
