"""Public synthetic regressions for the reviewed preprocessing failure modes."""
import pandas as pd
import pytest

from termmap.preprocessing.noise import NOISE_TYPES, corrupt
from termmap.preprocessing.normalize import Normalizer, clean, context_features, format_context


@pytest.mark.parametrize("mention,expected", [
    ("TSH <0.01 and T4 >2", "tsh <0.01 and t4 >2"),
    ("<b>HbA1c</b> >7%", "hba1c >7%"),
    (".5 mg", ".5 mg"),
    ("-3 mmol", "-3 mmol"),
    ("−3 mmol", "-3 mmol"),
    ("Pain.", "pain"),
    ("type 2.", "type 2"),
    ("HER2+", "her2+"),
    ("HER2-", "her2-"),
    ("ER-", "er-"),
    ("- chest pain -", "chest pain"),
    ("left grade III type 2 disease", "left grade iii type 2 disease"),
])
def test_cleanup_preserves_clinical_symbols(mention, expected):
    assert clean(mention) == expected


def test_context_requires_a_whole_mention_and_handles_whitespace():
    result = context_features("RA", "Radiology report. Assessment: RA is stable.")
    assert "[M]RA[/M]" in result["window"]
    assert "[M]Ra[/M]diology" not in result["window"]
    assert context_features("RA", "Radiology report.")["no_context"]
    spaced = context_features("chest pain", "Assessment: chest\n  pain is present.")
    assert not spaced["no_context"]
    assert "[M]chest\n  pain[/M]" in spaced["window"]


def test_repeated_mentions_report_ambiguity_without_using_assertion_polarity():
    result = context_features("pain", "ROS: no pain.\nAssessment: pain persists.")
    assert result["occurrence_count"] == 2
    assert result["ambiguous_context"]
    assert "assessment" in result["window"]
    assert not result["cues"]["negation"]
    negated = context_features("pain", "ROS: pain.\nAssessment: no pain.")
    assert negated["cues"]["negation"] == "no"


def test_missing_context_keeps_mention_cues_and_dictionary_codes():
    result = context_features("no chronic left pain I10.", "", ["I10"])
    assert result["no_context"]
    assert result["cues"]["negation"] == "no"
    assert result["cues"]["acuity"] == "chronic"
    assert result["cues"]["laterality"] == "left"
    assert result["literal_codes"] == ["I10"]
    assert context_features("I10.1234", "", ["I10"])["literal_codes"] == []
    assert context_features("QA0", "", ["QA0"])["literal_codes"] == ["QA0"]


def test_context_is_formatted_but_never_spell_corrected():
    chunk = "header_1: <b>Assessment:</b><br>TSH <0.01; patinet denies pain."
    formatted = format_context(chunk)
    assert "patinet" in formatted
    assert "<0.01" in formatted
    assert "<b>" not in formatted


def test_abbreviation_beam_retains_context_ranking():
    aliases = pd.DataFrame({"text": ["multiple sclerosis", "mitral stenosis"]})
    abbreviations = pd.DataFrame([
        dict(abbr="ms", expansion="multiple sclerosis", prior_count=2),
        dict(abbr="ms", expansion="mitral stenosis", prior_count=3),
    ])
    normalizer = Normalizer(aliases, abbreviations=abbreviations)
    views = normalizer.views("MS", "multiple sclerosis")
    assert views["V2"][0] == "multiple sclerosis"
    assert views["V0"] == ["MS"]


def test_spelling_is_nearest_dictionary_alternative_and_keeps_raw():
    aliases = pd.DataFrame({"text": ["diabetes mellitus", "cardiac disease", "scarring skin"]})
    normalizer = Normalizer(aliases)
    views = normalizer.views("diabtes mellitus")
    assert views["V3"] == ["diabetes mellitus"]
    assert views["V0"] == ["diabtes mellitus"]
    assert views["V1"] == ["diabtes mellitus"]
    assert normalizer.views("left stage diabtes")["V3"] == ["left stage diabtes"]
    assert normalizer.views("cardiac disease")["V3"] == ["cardiac disease"]


def test_spelling_rejects_unresolved_or_training_only_alternatives():
    aliases = pd.DataFrame({"text": ["cardiac heart", "cardial muscle", "cardiac", "cardial"]})
    train = pd.DataFrame([dict(entity="cardiaz",chunk_text="",display="",group_id="g1")])
    normalizer = Normalizer(aliases, train=train)
    assert normalizer.views("cardia")["V3"] == ["cardia"]
    assert normalizer.views("cardia", "heart")["V3"] == ["cardiac"]


def test_spelling_rejects_a_repaired_phrase_without_dictionary_support():
    normalizer = Normalizer(pd.DataFrame({"text": ["diabetes mellitus"]}))
    assert normalizer.views("unexpected diabtes problem")["V3"] == ["unexpected diabtes problem"]


@pytest.mark.parametrize("mention", ["glucoseu", "troponini", "troponint", "scooting", "beart"])
def test_spelling_preserves_field_suffixes_and_single_word_substitution(mention):
    normalizer = Normalizer(pd.DataFrame({"text": ["glucose", "troponin", "shooting", "heart"]}))
    assert normalizer.views(mention)["V3"] == [mention]


def test_anatomical_levels_and_alphanumeric_codes_are_not_expanded():
    abbreviations = pd.DataFrame([
        dict(abbr="s1", expansion="first heart sound", prior_count=10),
        dict(abbr="l5", expansion="fifth lumbar", prior_count=10),
    ])
    normalizer = Normalizer(pd.DataFrame({"text":["lumbar spine"]}), abbreviations=abbreviations)
    assert normalizer.views("l5 and s1 nerve root")["V2"] == ["l5 and s1 nerve root"]


@pytest.mark.parametrize("kind", NOISE_TYPES)
def test_synthetic_noise_does_not_change_qualifier_values(kind):
    noisy, _ = corrupt("stage advanced", kind=kind)
    assert "stage" in noisy.lower()
    assert "advanced" in noisy.lower()


def test_group_mask_applies_to_spelling_vocabulary():
    aliases = pd.DataFrame([
        dict(system="SNOMED",code="100",text="cardiac",norm_text="cardiac",source="abbr_variant"),
        dict(system="SNOMED",code="200",text="pain",norm_text="pain",source="dict_pref"),
    ])
    support = pd.DataFrame([dict(system="SNOMED",code="100",norm_text="cardiac",support_groups=["g1"])])
    normalizer = Normalizer(aliases, alias_support=support)
    assert normalizer.views("cardia", exclude_group="g1")["V3"] == ["cardia"]
    assert normalizer.views("cardia", exclude_group="g2")["V3"] == ["cardiac"]


def test_revised_query_caches_do_not_reuse_historical_rows(tmp_path):
    from termmap.retrieval.cache import directory
    cfg = {"root": tmp_path, "paths": {"index_dir": "old", "query_cache_dir": "reviewed"}}
    assert directory(cfg, "eval") == tmp_path / "reviewed" / "eval"
    assert not (tmp_path / "old").exists()
    del cfg["paths"]["query_cache_dir"]
    assert directory(cfg, "eval") == tmp_path / "old" / "eval"


def test_probe_aliases_are_excluded_from_fitting_without_changing_split():
    aliases = pd.DataFrame({"text": ["pain"]})
    train = pd.DataFrame([dict(entity="withheld", chunk_text="fittingonlytoken", display="pain", group_id="g1")])
    abbr = pd.DataFrame([dict(abbr="withheld", expansion="pain", prior_count=1)])
    normalizer = Normalizer(aliases, train, abbr, excluded_names={"withheld"})
    assert normalizer.vocab["fittingonlytoken"] == 0
    assert "withheld" not in normalizer.abbr
    assert len(train) == 1


def test_direct_retriever_applies_group_mask_and_system_specific_views():
    from termmap.retrieval.pipeline import Retriever
    seen = []
    class Encoder:
        def encode(self, texts):
            seen.append(list(texts))
            return texts
    class Search:
        allowed = None
        def __init__(self, code): self.code = code
        def search(self, texts, *args, **kwargs): return [[(self.code, 1.)] for _ in texts]
    class Views:
        def views(self, mention, context, exclude_group):
            assert exclude_group == "g1"
            assert "[M]" in context
            return {"V0":[mention], "V1":["base"], "V2":["expansion"],
                    "V3":["spelling"], "V4":["completion"]}
    retriever = object.__new__(Retriever)
    retriever.cfg = {"preprocessing":{"window_chars":300}, "retrieval":dict(
        channel_k=100, rrf_k=60, fused_k=50, pool_k=64, enabled_views=["V0","V1"],
        enabled_by_system={"ICD":["V0","V1","V2"], "SNOMED":["V0","V1","V2","V3"]})}
    retriever.icd = {"I10":{"parent":"", "children":[]}}
    retriever.normalizer = Views()
    retriever.encoder = Encoder()
    retriever.links = {}
    retriever.bridge = False
    for channel in ["lex", "word", "dense", "memdense"]:
        setattr(retriever, channel, {s:Search(c) for s,c in [("ICD","I10"),("SNOMED","100")]})
    retriever.exact = {"ICD":{}, "SNOMED":{}}
    retriever.retrieve_one("raw", "Assessment: raw", exclude_group="g1")
    assert seen == [["base","expansion","spelling"],["base","expansion"]]
