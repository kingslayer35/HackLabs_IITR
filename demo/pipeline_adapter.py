"""Warm adapter for the released F11 pipeline; imports its actual inference stages."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


class Pipeline:
    def __init__(self, pipeline_root, runtime, index_root=None, frozen_alias_root=None, vllm_python=None):
        root = Path(pipeline_root).resolve()
        sys.path.insert(0, str(root))
        from termmap.common.io import load_config, path_for, seed_everything, sha256, term_dir
        from termmap.common.release import verify_release
        from termmap.retrieval.online import OnlineRetriever
        from termmap.decision.layer import Features
        from termmap.reranking.hosted import HostedFinalRanker
        import joblib

        cfg = load_config(root / "configs/default.yaml")
        release = verify_release(cfg)  # Keep released settings/code/weights validation intact.
        self.flow = release["flow"]
        if self.flow != "F11":
            raise RuntimeError("The demo expects the released F11 inference flow.")
        if index_root:
            # Only paths change. Reuse the finalized research indexes after checking
            # the adapted checkpoint identities and train-only terminology tables.
            source = Path(index_root).resolve()
            metadata = json.loads((source / "eval/C2_v2/encoding_metadata.json").read_text())
            if metadata["profile"] != "eval" or not metadata["adapted"]:
                raise RuntimeError("The supplied index is not the selected adapted evaluation index.")
            if sha256(source / "eval/queries.json") != metadata["queries_sha256"]:
                raise RuntimeError("The search index query manifest changed.")
            source_terms = source.parent.parent / "terminology" / term_dir(cfg).name
            for table in term_dir(cfg).glob("*.parquet"):
                original = source_terms / table.name
                if original.exists() and sha256(table) != sha256(original):
                    raise RuntimeError("The released terminology differs from the index source: " + table.name)
            for system in ["ICD", "SNOMED"]:
                if sha256(path_for(cfg, "model_dir") / f"sapbert_v2/best_{system}/model.safetensors") != metadata["model_weights"][system]:
                    raise RuntimeError("Search index and released query encoder differ.")
                if sha256(source / f"eval/C2_v2/{system}_aliases.npy") != metadata["alias_vector_sha256"][system]:
                    raise RuntimeError("Frozen original alias vectors changed.")
            cfg["paths"]["query_cache_dir"] = str(source)
            # The curated checkout moved these same classes into subpackages.
            # Historic trusted lexical-index pickles still name the old module.
            import importlib
            sys.modules.setdefault("termmap.lexical", importlib.import_module("termmap.retrieval.lexical"))
            sys.modules.setdefault("termmap.retrieve", importlib.import_module("termmap.retrieval.lexical"))
        if frozen_alias_root:
            cfg["paths"]["frozen_alias_index_dir"] = str(Path(frozen_alias_root).resolve())
        cfg["paths"]["run_dir"] = str(Path(runtime).resolve() / "pipeline")
        cfg["paths"]["log_dir"] = str(Path(runtime).resolve() / "logs")
        if vllm_python:
            cfg["paths"]["vllm_python"] = str(Path(vllm_python).absolute())
        cfg["vllm"]["port"] = 19089  # A separate localhost service owned by this worker.
        cfg.setdefault("reranker", {})["cache_enabled"] = False
        self.cfg = cfg
        seed_everything(13, True)
        self.retriever = OnlineRetriever(cfg, self.flow)
        terminology = dict(icd=self.retriever.icd, sn=self.retriever.sn, aliases=self.retriever.aliases)
        adapter = path_for(cfg, "model_dir") / cfg["deployment"]["adapter_directory"]
        self.ranker = HostedFinalRanker(cfg, adapter, terminology=terminology)
        self.features = Features(cfg, self.retriever.icd, self.retriever.sn, cfg["deployment"]["profile"])
        self.model = joblib.load(path_for(cfg, "model_dir") / "decision/F11.joblib")
        self.release_sha256 = sha256(path_for(cfg, "model_dir") / "RELEASE.json")

    def predict(self, rows, folder, progress):
        import pandas as pd
        from termmap.decision.layer import apply_decision
        from termmap.evaluation.metrics import validate_predictions
        from .store import write_json

        tick = time.monotonic()
        data = pd.DataFrame(rows)
        progress("preparing", 0)
        prepared = self.retriever.prepare_inputs(rows, "clinical_entity")
        predictions, results, diagnostics = [], [], []
        for i, row in enumerate(rows):
            progress("retrieving", i)
            items = [self.retriever.retrieve(record) for record in prepared[i]]
            progress("reranking", i)
            scored = self.ranker.score_items(items)
            progress("deciding", i)
            matrices = [self.features.matrix(item) for item in scored]
            pools, confidence, missing = apply_decision(self.model, scored, matrices, self.features)
            output = dict(chunk_id=row["chunk_id"], clinical_entity=row["clinical_entity"])
            detail = dict(chunk_id=row["chunk_id"], clinical_entity=row["clinical_entity"], chunk_text=row["chunk_text"], systems={})
            for j, (item, pool) in enumerate(zip(scored, pools)):
                system = item["row"]["system"]
                code = str(pool[0][0])
                table = self.retriever.icd if system == "ICD" else self.retriever.sn
                output["icd10cm_code" if system == "ICD" else "snomed_code"] = code
                description = str(table[code].get("preferred", ""))
                diag = dict(system=system, code=code, description=description, confidence=float(confidence[j]),
                            low_confidence=bool(confidence[j] <= self.model["low_conf_threshold"]),
                            missing_probability=float(missing[j]),
                            alternatives=[dict(code=str(c), description=str(table[c].get("preferred", ""))) for c, _ in pool[:3]],
                            views=item["row"]["views"],
                            channels=[name for name, values in item["channels"].items() if code in dict(values)],
                            candidate_count=len(item["pool"]))
                detail["systems"][system] = diag
                diagnostics.append(dict(row=i, chunk_id=row["chunk_id"], **diag))
            predictions.append(output)
            results.append(detail)
            progress("deciding", i + 1)
        progress("validating", len(rows))
        frame = pd.DataFrame(predictions, columns=["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"])
        validate_predictions(data, frame, {"ICD": set(self.retriever.icd), "SNOMED": set(self.retriever.sn)}, self.retriever.icd, self.features.exceptions)
        frame.to_csv(folder / "predictions.csv", index=False)
        elapsed = time.monotonic() - tick
        write_json(folder / "results.json", dict(flow=self.flow, rows=results, elapsed=elapsed, validator="passed", release_sha256=self.release_sha256))
        write_json(folder / "diagnostics.json", diagnostics)
        return elapsed

    def close(self):
        self.ranker.close()
