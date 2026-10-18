import numpy as np
import pandas as pd
from termmap.retrieval.lexical import CharIndex


def test_targeted_alias_removal_equals_full_recomputation():
    # A held-out name must disappear globally, including from a second concept.
    a=pd.DataFrame(dict(code=["a","a","b","c"],text=["heart failure","cardiac failure","heart failure","renal failure"]))
    a["norm_text"]=a.text;index=CharIndex(a);texts=["heart failure","failure"]
    base=index.score_all(texts).max(0)
    fast=index.scores_without_alias(texts,"heart failure",base)
    full=index.score_all(texts,exclude_norm="heart failure").max(0)
    np.testing.assert_allclose(fast,full,atol=1e-6)
