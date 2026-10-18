"""Public synthetic cases; no row-specific correction tables."""
import pandas as pd

from termmap.preprocessing.normalize import Normalizer


def normalizer():
    aliases=pd.DataFrame([
        dict(system="SNOMED",code="1",text="alpha beta syndrome",norm_text="alpha beta syndrome",source="dict_pref"),
        dict(system="SNOMED",code="2",text="arterial blockade",norm_text="arterial blockade",source="dict_pref"),
        dict(system="SNOMED",code="3",text="x-ray examination",norm_text="x-ray examination",source="dict_pref"),
    ])
    abbreviations=pd.DataFrame([
        dict(abbr="ab",expansion="alpha beta syndrome",prior_count=1),
        dict(abbr="ab",expansion="arterial blockade",prior_count=100),
        dict(abbr="got",expansion="glutamate oxaloacetate transaminase",prior_count=1),
        dict(abbr="x-ray",expansion="electromagnetic radiation",prior_count=1),
    ])
    result=Normalizer(aliases,abbreviations=abbreviations)
    result.tightened=True
    return result


def test_ambiguous_acronym_is_preserved_without_context_evidence():
    norm=normalizer()
    assert norm.views("AB")["V2"]==["ab"]
    assert norm.abbreviation_diagnostics("AB")[0]["status"]=="unresolved"


def test_generic_clinical_words_cannot_disambiguate_an_acronym():
    assert normalizer().views("AB", "patient history of a chronic syndrome")["V2"]==["ab"]


def test_evidence_beats_a_more_frequent_unrelated_sense():
    assert normalizer().views("AB", "alpha beta symptoms")["V2"]==["alpha beta syndrome"]


def test_explicit_local_definition_selects_only_supported_meaning():
    assert normalizer().views("AB", "alpha beta syndrome (AB)")["V2"]==["alpha beta syndrome"]


def test_lowercase_ordinary_word_is_preserved_despite_inventory_entry():
    assert normalizer().views("patient got better")["V2"]==["patient got better"]


def test_existing_hyphenated_medical_word_is_not_expanded_to_its_definition():
    assert normalizer().views("x-ray report")["V2"]==["x-ray report"]


def test_public_code_identity_allows_same_sense_orthographic_forms():
    aliases=pd.DataFrame([
        dict(system="SNOMED",code="1",text=x,norm_text=x,source="dict_pref")
        for x in ["alpha beta condition","alpha-beta condition"]])
    abbr=pd.DataFrame([dict(abbr="abc",expansion=x,prior_count=1)
                       for x in aliases.text])
    norm=Normalizer(aliases,abbreviations=abbr)
    assert set(norm.views("ABC","alpha beta condition")["V2"])==set(aliases.text)


def test_self_group_sense_cannot_be_rescued_by_context():
    aliases=pd.DataFrame({"text":["alpha beta"]})
    train=pd.DataFrame([dict(entity="AB",chunk_text="alpha beta",display="Alpha beta",group_id="g1")])
    abbr=pd.DataFrame([dict(abbr="ab",expansion="alpha beta",prior_count=1)])
    norm=Normalizer(aliases,train,abbr)
    assert norm.views("AB","alpha beta",exclude_group="g1")["V2"]==["ab"]
    assert norm.views("AB","alpha beta",exclude_group="g2")["V2"]==["alpha beta"]


def test_inventory_paraphrase_cannot_introduce_new_negation():
    aliases=pd.DataFrame({"text":["electrocardiogram", "no new changes"]})
    abbr=pd.DataFrame([dict(abbr="ecg",expansion=e,prior_count=n)
                       for e,n in [("no new changes",100),("electrocardiogram",1)]])
    norm=Normalizer(aliases,abbreviations=abbr)
    assert "no new changes" not in norm.views("ECG", "no new changes on review")["V2"]


def test_public_dictionary_can_authorize_nonliteral_spelling():
    aliases=pd.DataFrame({"text":["EKG - electrocardiogram"]})
    abbr=pd.DataFrame([dict(abbr="ekg",expansion="electrocardiogram",prior_count=1)])
    assert Normalizer(aliases,abbreviations=abbr).views("EKG","electrocardiogram")["V2"]==["electrocardiogram"]


def test_one_inventory_meaning_without_context_is_still_not_proof():
    aliases=pd.DataFrame({"text":["nuclear cytoplasmic"]})
    abbr=pd.DataFrame([dict(abbr="n/c",expansion="nuclear cytoplasmic",prior_count=1)])
    assert Normalizer(aliases,abbreviations=abbr).views("n/c")["V2"]==["n/c"]


def test_plural_dictionary_words_are_not_split_into_unrelated_fragments():
    aliases=pd.DataFrame({"text":["readmission", "read missions", "suture passer", "pass ers"]})
    norm=Normalizer(aliases)
    assert norm.views("readmissions risk")["V4"]==["readmissions risk"]
    assert norm.views("suture passers")["V4"]==["suture passers"]


def test_split_requires_a_known_phrase_not_merely_two_known_fragments():
    aliases=pd.DataFrame({"text":["ration", "ale", "back pain"]})
    norm=Normalizer(aliases)
    assert norm.views("rationale")["V4"]==["rationale"]
    assert norm.views("new backpain")["V4"]==["new back pain"]


def test_long_reports_receive_cleanup_without_speculative_views():
    norm=normalizer()
    text="AB patient got better. "*20
    views=norm.views(text,"alpha beta syndrome")
    assert views["V0"]==[text]
    assert views["V2"]==views["V1"]==views["V3"]==views["V4"]
    assert norm.abbreviation_diagnostics(text)[0]["reason"]=="extended_text_span"


def test_a_single_shared_word_cannot_select_an_unrelated_reading():
    assert normalizer().views("AB", "alpha symptoms")["V2"]==["ab"]


def test_single_letter_context_is_not_support_for_an_expansion():
    aliases=pd.DataFrame({"text":["protein c inhibitor"]})
    abbr=pd.DataFrame([dict(abbr="pci",expansion="protein c inhibitor",prior_count=1)])
    assert Normalizer(aliases,abbreviations=abbr).views("PCI", "vitamin C, plan followup")["V2"]==["pci"]


def test_local_definitions_and_following_words_are_not_duplicated():
    norm=normalizer()
    assert norm.views("AB (alpha beta syndrome)")["V2"]==["alpha beta syndrome"]
    # A parenthesized acronym can stay intact when its full form is present.
    assert norm.views("alpha beta syndrome (AB)")["V2"]==["alpha beta syndrome (ab)"]
    assert norm.views("AB syndrome", "alpha beta syndrome")["V2"]==["alpha beta syndrome"]
    assert norm.views("alpha beta syndrome AB")["V2"]==["alpha beta syndrome"]
