from termmap.preprocessing.normalize import clean,context_features,Normalizer
from termmap.preprocessing.noise import corrupt,NOISE_TYPES
import pandas as pd

def test_protected_tokens():
    aliases=pd.DataFrame({"text":["left knee pain", "type 2 diabetes", "no chronic disease"]})
    normalizer=Normalizer(aliases)
    for mention in ["left knee pain", "type 2 diabetes", "no chronic disease"]:
        views=normalizer.views(mention)
        for key in ["V1","V2","V3","V4"]:assert views[key]==[mention]
    for kind in NOISE_TYPES:
        noisy,_=corrupt("no acute left type 2",kind)
        assert "2" in noisy and "left" in noisy.lower() and "no" in noisy.lower()

def test_clean_and_context():
    assert clean("<br>CHF exacerbation (HCC) !!")=="chf exacerbation"
    assert clean("(pain")=="pain"
    context=context_features("CHF","PMH: no CHF. hypercapnea",["I50.9"])
    assert "[M]CHF[/M]" in context["window"]
    assert context["cues"]["negation"]=="no"
    assert context["cues"]["history"]=="pmh"
    assert context_features("missing","text")["no_context"]
