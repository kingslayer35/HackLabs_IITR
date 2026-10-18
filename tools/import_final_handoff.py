"""Import completed final reports/decisions without evaluating or opening TEST.

Requires the main evaluation's final readiness marker and passing validators.
Only this submission checkout is written. Run once final results are ready.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

FLOWS = ("F00", "F10", "F01", "F11")
REPORTS = ("README.md", "Final_all_flows_DEV_TEST.md", "Final_all_flows_metrics.csv",
           "Final_all_flows_full_metrics.json", "Judging_metrics.json")


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(2**20),b""):value.update(block)
    return value.hexdigest()


def portable_report(text):
    """Preserve all metric tables while removing research-only navigation."""
    text=re.sub(r"\[([^\]]+)\]\(\.\./artifacts/runs/[^/]+/(F(?:00|10|01|11))/([^)]*)\)",
        lambda m:f"{m[1]}: `private_assets/evaluation/{m[2]}/{m[3]}`",text)
    text=text.replace("[problem statement](../../TermMap_Problem_Statement_Explained.md)",
                      "[submission requirements](../docs/DELIVERABLES.md)")
    historical=("Four_flow_DEV_earlier_LoRA_vllm.md","Epoch_comparison_revised_LoRA.md",
                "Four_flow_DEV_revised_epoch2_vllm.md","Final_DEV_TEST_F00_F11.md")
    text="\n".join(line for line in text.splitlines() if not any(name in line for name in historical))+"\n"
    text=text.replace("The historical snapshots retain their original model/protocol labels; current final results are the four-flow epoch-2 tables above.",
        "Historical snapshots remain in the research repository. This submission contains the authoritative final epoch-2 four-flow tables and aggregate exports.")
    text+="\nPer-input predictions, diagnostics and validation files named above are distributed only in the restricted private bundle. See [results provenance and private-file access](../docs/RESULTS_PROVENANCE.md).\n"
    return text


def main():
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);args=parser.parse_args()
    import yaml
    source=args.source.resolve();target=Path(__file__).resolve().parents[1]
    def config(name):
        file=source/"configs"/name
        value=yaml.safe_load(file.read_text());value["root"]=source
        return value
    def path(cfg,name):
        p=Path(cfg["paths"][name]);return p if p.is_absolute() else cfg["root"]/p
    base=config("vllm_original.yaml");epoch2=config("revised_epoch2.yaml")
    test_owners={"F00":base,"F10":config("f10_test_part0.yaml"),
                 "F01":config("f01_test_extension.yaml"),"F11":config("f11_test_gpu2.yaml")}
    base_root=path(base,"run_dir");epoch2_root=path(epoch2,"run_dir")
    final=json.loads((base_root/"requested_tests/FINAL_RESULTS_READY.json").read_text())
    if not final.get("scope_complete") or final["test_flows"]!=["F00","F11"]:
        raise RuntimeError("The requested main-run evaluation is not complete")
    frozen=json.loads((epoch2_root/"frozen/EPOCH2_DEV_FROZEN.json").read_text())
    if frozen["selected_epoch"]!=2:raise RuntimeError("Expected the explicitly selected epoch 2")
    aggregate=json.loads((source/"results/Final_all_flows_full_metrics.json").read_text())
    for dataset in ["DEV","PROBE","TEST"]:
        if set(aggregate["metrics"][dataset])!=set(FLOWS):
            raise RuntimeError("The final aggregate is missing a completed flow")
    private=target/"private_assets";models=private/"models"
    evidence={};copies=[];decisions={}
    if sha(models/"reranker_lora/epoch_2/adapter_model.safetensors")!=frozen["adapter_sha256"]:
        raise RuntimeError("The packaged adapter differs from the evaluated epoch 2")
    for flow in FLOWS:
        owner=base if flow in ["F00","F10"] else epoch2
        decision=path(owner,"model_dir")/"decision"/(flow+".joblib")
        if sha(decision)!=frozen["decisions_sha256"][flow]:
            raise RuntimeError("A frozen decision model changed")
        decisions[flow]=decision
        for dataset in ["dev","probe","test"]:
            scorer=test_owners[flow] if dataset=="test" else owner
            folder=path(scorer,"run_dir")/flow
            validation=json.loads((folder/(dataset+"_validation.json")).read_text())
            if not validation["passed"] or validation["auxiliary_labels_used"]:
                raise RuntimeError("A final prediction validator has not passed")
            if dataset=="dev" and not validation["held_out_group_isolation"]:
                raise RuntimeError("DEV group isolation was not validated")
            if sha(folder/(dataset+"_predictions.csv"))!=validation["predictions_sha256"]:
                raise RuntimeError("Saved predictions differ from their validated bytes")
            metrics=folder/(dataset+"_metrics.json")
            if json.loads(metrics.read_text())!=aggregate["metrics"][dataset.upper()][flow]["3"]:
                raise RuntimeError("Published aggregates differ from validated final metrics")
            if dataset=="dev" and sha(metrics)!=frozen["flow_metrics_sha256"][flow]:
                raise RuntimeError("DEV metrics changed after the decision freeze")
            copies.append((flow,dataset,folder))
            evidence[flow+"/"+dataset]=dict(validation_sha256=sha(folder/(dataset+"_validation.json")),
                metrics_sha256=sha(metrics),predictions_sha256=validation["predictions_sha256"],
                rows=validation["rows"],passed=True,auxiliary_labels_used=False,
                decision_protocol=validation["decision_protocol"])
    # All twelve validators and frozen models pass before any artifact is copied.
    for flow,dataset,folder in copies:
        destination=private/"evaluation"/flow;destination.mkdir(parents=True,exist_ok=True)
        for suffix in ["metrics.json","raw_metrics.json","predictions.csv","details.parquet",
                       "validation.json","diagnostics.jsonl","prediction_rows.parquet"]:
            name=dataset+"_"+suffix
            if (folder/name).is_file():shutil.copyfile(folder/name,destination/name)
    for flow,decision in decisions.items():
        (models/"decision").mkdir(parents=True,exist_ok=True)
        shutil.copyfile(decision,models/"decision"/decision.name)
    classifier=path(base,"model_dir")/("type_eval_"+base["generation"]+"_C1.joblib")
    deployment=yaml.safe_load((target/"configs/default.yaml").read_text())
    shutil.copyfile(classifier,models/("type_eval_"+deployment["generation"]+"_C1.joblib"))
    if classifier.with_suffix(".json").is_file():
        shutil.copyfile(classifier.with_suffix(".json"),models/("type_eval_"+deployment["generation"]+"_C1.json"))
    confirmation=json.loads((epoch2_root/"C2_v2/confirmation.json").read_text())
    for system in ["ICD","SNOMED"]:
        old=Path(confirmation["effective_models"][system])
        effective=models/"sapbert_v2"/("best_"+system) if "best_" in old.name else models/"base_sapbert"
        confirmation["effective_models"][system]=str(effective.relative_to(target))
    run=target/deployment["paths"]["run_dir"]/"C2_v2";run.mkdir(parents=True,exist_ok=True)
    (run/"confirmation.json").write_text(json.dumps(confirmation,indent=2)+"\n")
    reports=[]
    for name in REPORTS:
        original=source/"results"/name;output=target/"results"/name
        if original.suffix==".md":
            text=original.read_text();portable=portable_report(text)
            if [line for line in text.splitlines() if line.startswith("|")]!=[line for line in portable.splitlines() if line.startswith("|")]:
                raise RuntimeError("A published metric table changed during packaging")
            output.write_text(portable)
        else:shutil.copyfile(original,output)
        reports.append(dict(file="results/"+name,source_sha256=sha(original),packaged_sha256=sha(output),
                            metric_tables_unchanged=True))
    with (target/"results/Final_all_flows_metrics.csv").open() as stream:
        records=sum(1 for _ in csv.DictReader(stream))
    publication=dict(imported_utc=datetime.now(timezone.utc).isoformat(),source_project="HackLabs",
        sets=["DEV","PROBE","TEST"],flows=list(FLOWS),selected_lora_epoch=2,
        aggregate_csv_records=records,source_modified=False,evaluation_rerun=False,
        raw_sealed_partition_opened=False,private_data_committed=False,
        transformations=["Research-only links replaced with private-bundle paths", "Historical snapshot navigation omitted; final tables preserved"],
        reports=reports,validated_outputs=evidence,decisions_sha256=frozen["decisions_sha256"])
    (target/"results/MANIFEST.json").write_text(json.dumps(publication,indent=2)+"\n")
    manifest=json.loads((private/"MANIFEST.json").read_text())
    eligible=[]
    for folder in [target/"termmap",target/"scripts",private/"models",private/"terminology",private/"data"]:
        eligible.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                        and p.name!="RELEASE.json")
    eligible.extend((private/"resources"/name) for name in ["model_snapshot.json","reranker_snapshot.json"])
    release=dict(complete=True,flow="F11",dictionary_hash=manifest["dictionary_hash"],
        settings={key:deployment[key] for key in ["retrieval","preprocessing","reranker","decision","deployment"]},
        files={str(p.relative_to(target)):sha(p) for p in sorted(eligible)},
        source_validations=evidence,trained_on_test=False,source_selected_epoch=2,
        deployment_gpu_acceptance="Run separately on the target GPU before submission")
    (models/"RELEASE.json").write_text(json.dumps(release,indent=2)+"\n")
    manifest["deployment_decision_layers"]="Imported frozen complete DEV fits; TEST did not refit them"
    (private/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Imported all four DEV/PROBE/TEST flows, twelve validated outputs and frozen decisions.")
    print("Published aggregate records:",records,"Repackage assets and run GPU deployment acceptance.")


if __name__=="__main__":main()
