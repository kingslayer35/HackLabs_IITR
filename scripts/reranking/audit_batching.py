"""Check concurrent full-pool repeat/shuffle invariance without using gold labels."""
import json
import time

import joblib
import numpy as np

from termmap.common.io import load_config, path_for, write_json
from termmap.reranking.vllm import VllmRanker
from termmap.reranking.client import make_ranker


def main():
    cfg=load_config();root=path_for(cfg,"run_dir");cards=None;result={}
    for adapter in [True,False]:
        ranker=make_ranker(cfg,adapter=adapter,cards=cards);cards=ranker.cards
        payloads=[]
        for source in ["C1","C2_v2"]:
            for item in joblib.load(root/"vllm_service"/(source+"_audit.joblib")):
                for value in [item,item["other_system"]]:
                    payloads.extend(cards.payload(value["row"],c,seen=cfg["reranker"].get("seen_enabled",True)) for c,_ in value["pool"])
        tick=time.perf_counter()
        assert ranker.token_ids(payloads)==VllmRanker.token_ids(ranker,payloads), "Input cache changed the packed tokens"
        original=ranker.score(payloads,use_cache=False)
        first_seconds=time.perf_counter()-tick
        repeated=ranker.score(payloads,use_cache=False)
        shuffled=ranker.score(list(reversed(payloads)),use_cache=False)[::-1]
        repeated_error=float(np.max(np.abs(np.array(original)-repeated)))
        shuffled_error=float(np.max(np.abs(np.array(original)-shuffled)))
        result["lora" if adapter else "zero"]=dict(
            identity=ranker.identity,candidate_pairs=len(payloads),
            first_pass_seconds=first_seconds,
            repeated_max_absolute_difference=repeated_error,
            shuffled_max_absolute_difference=shuffled_error,
            passed=original==repeated==shuffled,
            workers=cfg["vllm"]["workers"],request_batch=cfg["vllm"]["request_batch"],
            cache_disabled=True,gold_labels_used=False,exact_reference_tokens=True)
        ranker.close()
    passed=all(v["passed"] for v in result.values())
    write_json(root/"vllm_service/BATCH_AUDIT.json",dict(passed=passed,roles=result,
        exact_tolerance="Scores rounded to 1e-5; exact equality across repeated/reversed concurrent requests",
        test_opened=False))
    print("Full-pool concurrent batching audit",json.dumps(result),flush=True)
    if not passed:raise RuntimeError("The new backend fails full-pool request-order determinism")


if __name__=="__main__":main()
