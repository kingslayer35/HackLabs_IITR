import hashlib
import re
import sqlite3
import numpy as np

from termmap.preprocessing.normalize import context_features
from termmap.reranking.prompts import pack_payload
from termmap.reranking.model import Reranker


class Tokens:
    def encode(self,text,add_special_tokens=False):
        return [m.group() for m in re.finditer(r"\S+",text)]
    def __call__(self,text,**kwargs):
        spans=[m.span() for m in re.finditer(r"\S+",text)]
        return dict(input_ids=list(range(len(spans))),offset_mapping=spans)


def long_payload():
    return ("<Instruct>: Match the highlighted entity exactly.\n<Query>: Type: DiagnosisCode\nMention: hypertension\nContext: "
            +"earlier statement "*250+"[M]hypertension[/M]"+" later statement"*250
            +"\n<Document>: I10 | Essential hypertension | also: "+"public synonym "*200)


def test_budget_preserves_target_and_candidate_and_fitting_inputs():
    tokenizer=Tokens();text=long_payload();packed=pack_payload(tokenizer,text,160)
    assert len(tokenizer.encode(packed))<=160
    assert "[M]hypertension[/M]" in packed
    assert "<Document>: I10 | Essential hypertension" in packed
    assert packed.startswith("<Instruct>: Match the highlighted entity exactly.")
    assert pack_payload(tokenizer,packed,160)==packed
    small="<Instruct>: Match.\n<Query>: hypertension\n<Document>: I10 | Hypertension"
    assert pack_payload(tokenizer,small,160)==small


def test_cached_score_must_describe_the_actual_packed_input():
    tokenizer=Tokens();original=long_payload();ranker=object.__new__(Reranker)
    ranker.tokenizer=tokenizer;ranker.prefix=[];ranker.suffix=[];ranker.cfg={}
    ranker.connection=sqlite3.connect(":memory:")
    ranker.connection.execute("CREATE TABLE scores (key TEXT PRIMARY KEY,score REAL NOT NULL)")
    key=hashlib.sha256(original.encode()).hexdigest()
    ranker.connection.execute("INSERT INTO scores VALUES (?,?)",(key,-999.))
    observed=[]
    def logits(payloads):
        import torch
        observed.extend(payloads)
        return torch.tensor([7.]*len(payloads))
    ranker.logits=logits
    assert ranker.score([original])==[7.]
    assert observed[0]!=original and "<Document>: I10" in observed[0]
    assert ranker.score([original])==[7.] and len(observed)==1
    ranker.connection.close()


def test_context_highlights_a_complete_mention_before_a_substring():
    result=context_features("CT","Subjective: no pain. Plan: obtain CT of the chest.")
    assert "[M]CT[/M]" in result["window"]
    assert "subje[M]ct[/M]ive" not in result["window"].lower()
    assert not result["no_context"]
