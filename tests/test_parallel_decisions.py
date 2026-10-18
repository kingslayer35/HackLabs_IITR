import json
from scripts.decision.parallel import ready
from scripts.decision.fit import matching_fit
from termmap.common.io import sha256


def test_parallel_decisions_reject_copied_timings_until_current_gpu_finishes(tmp_path):
    flow=tmp_path/"F10";flow.mkdir()
    for name in ["dev_timing.json","probe_timing.json"]:(flow/name).write_text("{}")
    jobs=tmp_path/"jobs/revised_flows";jobs.mkdir(parents=True)
    (jobs/"JOB_ID").write_text("123")
    assert not ready(tmp_path,"F10","revised_flows")
    (flow/"SCORE_COMPLETE.json").write_text(json.dumps(dict(slurm_job_id="122")))
    assert not ready(tmp_path,"F10","revised_flows")
    (flow/"SCORE_COMPLETE.json").write_text(json.dumps(dict(slurm_job_id="123")))
    assert ready(tmp_path,"F10","revised_flows")


def test_parallel_worker_does_not_reuse_a_fit_after_model_or_input_drift(tmp_path):
    model=tmp_path/"model.joblib";model.write_bytes(b"original-model")
    marker=tmp_path/"complete.json";expected=dict(inputs_sha256="scores-v1")
    marker.write_text(json.dumps(dict(provenance=expected,outputs={str(model):sha256(model)})))
    assert matching_fit(marker,expected)
    assert not matching_fit(marker,dict(inputs_sha256="scores-v2"))
    model.write_bytes(b"replaced-model")
    assert not matching_fit(marker,expected)
    marker.write_text("incomplete write")
    assert not matching_fit(marker,expected)
