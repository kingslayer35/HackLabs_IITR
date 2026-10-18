"""Memoization must preserve held-alias and leave-group-out card text."""
from collections import defaultdict
from functools import lru_cache

from termmap.reranking.cards import CachedCards
from termmap.reranking.model import Cards


def test_cached_cards_equal_reference_across_alias_group_seen_and_system_masks():
    reference=object.__new__(Cards)
    reference.held={"held out"}
    reference.aliases=defaultdict(list,{("ICD","I10"): ["Hypertension","held out","HTN"],
                                     ("SNOMED","38341003"): ["Hypertension","held out","HTN"]})
    reference.icd={"I10":dict(preferred="Hypertension",parent="",category="I10",is_leaf=True,
                              refinement_text="",seventh_char_meaning="")}
    reference.sn={"38341003":dict(preferred="Hypertension",fsn="Hypertension (disorder)",
                                  tag="disorder",top_level="Clinical finding")}
    reference.self_only_names=defaultdict(set,{("ICD","I10","self"): {"htn"},
                                              ("SNOMED","38341003","self"): {"htn"}})
    reference.examples=defaultdict(list,{("ICD","I10"): [("HTN","self"),("raised pressure","other")],
                                        ("SNOMED","38341003"): [("HTN","self"),("raised pressure","other")]})
    cached=object.__new__(CachedCards)
    cached.__dict__.update(reference.__dict__)
    cached._cached_card=lru_cache(maxsize=65536)(cached._make_card)
    for system,code in [("ICD","I10"),("SNOMED","38341003")]:
        for alias in [None,"hypertension","htn"]:
            for group in [None,"self","other"]:
                for seen in [False,True]:
                    row=dict(system=system,excluded_alias=alias,excluded_group=group)
                    expected=reference.card(row,code,seen=seen)
                    assert cached.card(row,code,seen=seen)==expected
                    assert cached.card(row,code,seen=seen)==expected
                    assert "held out" not in expected
    assert cached._cached_card.cache_info().hits==36
