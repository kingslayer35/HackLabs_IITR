import pandas as pd
import pytest
from termmap.common.io import load_config,term_dir
from termmap.terminology.hierarchy import semantic_tag,core_words

@pytest.fixture(scope="module")
def tables():
    out=term_dir(load_config())
    return (pd.read_parquet(out/"icd_concepts.parquet").set_index("code"),
            pd.read_parquet(out/"snomed_concepts.parquet").set_index("code"))

def test_icd_tree(tables):
    icd,_=tables
    assert set(icd.loc["N18.3","children"])=={"N18.30","N18.31","N18.32"}
    assert set(icd.loc["I50.2","children"])=={"I50.20","I50.21","I50.22","I50.23"}
    code="T36.0X1A"
    while icd.loc[code,"parent"]:code=icd.loc[code,"parent"]
    assert code=="T36"
    assert len(icd)==98182
    assert icd.is_leaf.sum()==74718
    assert "QA0.0101" in icd.index

def test_snomed_tags_and_links(tables):
    _,sn=tables
    assert len(sn)==382124
    assert sn.loc["38341003","tag"]=="disorder"
    assert sn.loc["38341003","top_level"]=="Clinical finding"
    assert sn.loc["255604002","tag"]=="qualifier value"
    assert sn.loc["255604002","top_level"]=="Qualifier value"
    assert sn.tag.ne("unknown").sum()==382120
    links=pd.read_parquet(term_dir(load_config())/"snomed_links.parquet")
    mask=links.src.isin(["433144002","700378005"]) & links.dst.isin(["433144002","700378005"])
    assert mask.any()

def test_only_official_tags():
    assert semantic_tag("Noun (left ventricle)")==""
    assert semantic_tag("Noun (disorder)")=="disorder"
    assert core_words("Pain in left knee")==core_words("Knee pain")

def test_all_parent_links_keep_same_tag(tables):
    _,sn=tables
    links=pd.read_parquet(term_dir(load_config())/"snomed_links.parquet")
    tags=sn.tag.to_dict()
    same=(links.src.map(tags)==links.dst.map(tags))
    assert (same | links.kind.eq("qualifier_sibling")).all()

def test_tree_intervals_and_official_meanings(tables):
    icd,_=tables
    for code,n in [("N18",11),("I50",29)]:
        row=icd.loc[code]
        assert row.pre_out-row.pre_in+1==n
    assert icd.loc["T36.0X1A","seventh_char_meaning"]=="initial encounter"
    assert icd.loc["S52.521","refinement_kind"]=="laterality"

def test_trusted_specifier_and_context_links(tables):
    links=pd.read_parquet(term_dir(load_config())/"snomed_links.parquet")
    edges=set(zip(links.src,links.dst,links.kind))
    for a,b in [("709044004","433144002"),("433144002","700378005"),("433144002","700379002")]:
        assert (a,b,"specifier_child") in edges
    assert ("38341003","70995007","specifier_child") not in edges
    assert ("275526006","230690007","qualifier_sibling") in edges
