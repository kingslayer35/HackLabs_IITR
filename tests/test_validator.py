import pandas as pd
import pytest
from termmap.evaluation.metrics import validate_predictions

def sample():
    inputs=pd.DataFrame({"chunk_id":["c"],"clinical_entity":["x"]})
    predictions=inputs.assign(icd10cm_code=["I10"],snomed_code=["123456789012345678"])
    return inputs,predictions,{"ICD":{"I10"},"SNOMED":{"123456789012345678"}}

def test_string_ids():
    i,p,d=sample();validate_predictions(i,p,d)
    p["snomed_code"]=[123456789012345678]
    with pytest.raises(ValueError):validate_predictions(i,p,d)

def test_dictionary_membership_and_order():
    i,p,d=sample();p["icd10cm_code"]=["INVALID"]
    with pytest.raises(ValueError):validate_predictions(i,p,d)
    i,p,d=sample();p["clinical_entity"]=["changed"]
    with pytest.raises(ValueError):validate_predictions(i,p,d)
