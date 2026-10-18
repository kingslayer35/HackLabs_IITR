"""Memoize unchanged candidate cards without weakening alias/group masks."""
from functools import lru_cache

from termmap.reranking.model import Cards


class CachedCards(Cards):
    def __init__(self, cfg, profile="eval", terminology=None):
        super().__init__(cfg, profile=profile, terminology=terminology)
        self._cached_card = lru_cache(maxsize=65536)(self._make_card)

    def _make_card(self, system, code, excluded_alias, excluded_group, seen):
        return super().card(dict(system=system, excluded_alias=excluded_alias,
                                 excluded_group=excluded_group), code, seen=seen)

    def card(self, row, code, seen=True):
        return self._cached_card(row["system"], code, row.get("excluded_alias"),
                                 row.get("excluded_group"), seen)
