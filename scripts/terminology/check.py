"""Evaluate P0 gates without reopening data or rerunning experiments."""
import json
from termmap.common.io import load_config,term_dir,path_for,write_json

cfg=load_config();out=term_dir(cfg);expected=cfg["gates"]
audit=json.loads((out/"terminology_audit.json").read_text())
metrics=json.loads((path_for(cfg,"run_dir")/"C0-ref"/"metrics.json").read_text())
checks={
    "icd_codes":audit["icd"]["code_count"]==expected["icd_codes"],
    "icd_leaves":audit["icd"]["leaves"]==expected["icd_leaves"],
    "snomed_concepts":audit["snomed"]["concepts"]==expected["snomed_concepts"],
    "snomed_tagged":audit["snomed"]["tagged_concepts"]==expected["snomed_tagged"],
    "icd_gold_coverage":audit["split"]["gold_coverage"]["ICD"]>=.999 and audit["split"]["invalid_gold"]["ICD"]==["L6.0"],
    "snomed_gold_coverage":audit["split"]["gold_coverage"]["SNOMED"]==1,
    "icd_crosscheck":audit["simple_icd_crosscheck"]["parent_mismatches"]==0,
    "allergy_labels_present":audit["allergy_labels"]["all_present_in_main_concept_table"],
}
differences={}
for system,values in expected["c0_reference"].items():
    for metric,reference in values.items():
        measured=metrics[system][metric];difference=measured-reference
        key=f"{system}/{metric}"
        differences[key]=dict(reference=reference,measured=measured,difference_points=difference)
        checks[f"C0/{key}"]=abs(difference)<=expected["c0_tolerance_points"]
result=dict(passed=all(checks.values()),checks=checks,baseline_comparison=differences,
            note="Unit test results are recorded separately in logs/tests_P0.log; all must pass before phase completion.")
write_json(path_for(cfg,"run_dir")/"P0"/"gate_results.json",result)
text="\n## P0 exit gate\n\n"
text+="v2.1 validates the isolated C0-ref harness here; official dev C0 remains unchanged. Gate failures use the full-run fallback policy.\n\n"
text+="| System | Metric | Reference | Measured | Difference (points) | Within ±0.5 |\n|---|---|---:|---:|---:|---|\n"
for key,v in differences.items():
    system,metric=key.split("/")
    text+=f"| {system} | {metric} | {v['reference']:.3f} | {v['measured']:.3f} | {v['difference_points']:+.3f} | {checks['C0/'+key]} |\n"
text+="\nC0-ref uses the exact v2.1 recipe: char_wb (3,4), min_df=2, sublinear TF, normalized dictionary descriptions only, max over names, all 2,306 ICD and 2,700 SNOMED-Dx rows. This isolated reference feeds no flow indexes or training artifacts. The earlier concatenated-document attempt is historical and rejected.\n"
report=path_for(cfg,"report_dir")/"checkpoints.md"
existing=report.read_text().split("\n## P0 exit gate")[0]
report.write_text(existing+text)
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["passed"] else 1)
