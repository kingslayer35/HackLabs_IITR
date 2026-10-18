"""Keep public synthetic tests usable before the private ZIP is restored."""
from pathlib import Path
import pytest


PRIVATE_TESTS = {
    "test_icd_tree", "test_snomed_tags_and_links", "test_all_parent_links_keep_same_tag",
    "test_tree_intervals_and_official_meanings", "test_trusted_specifier_and_context_links",
    "test_split_and_probe_isolation", "test_preserved_icd_subconcept",
    "test_training_tables_and_probe_exclusions",
}


def pytest_collection_modifyitems(items):
    root = Path(__file__).resolve().parents[1]
    if not (root / "private_assets/data/icd10cm_codes.xlsx").exists():
        for item in items:
            if item.name in PRIVATE_TESTS:
                item.add_marker(pytest.mark.skip(reason="Restore the private artifact ZIP for dictionary integration tests"))
