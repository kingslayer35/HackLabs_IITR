"""Local Qwen3 0.6B cards, official prompt, yes-minus-no scoring and LoRA."""
from __future__ import annotations
from collections import defaultdict,Counter
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel,LoraConfig,get_peft_model
from termmap.common.io import term_dir,path_for,sha256,seed_everything,write_json
from termmap.terminology.hierarchy import strip_tag
from termmap.preprocessing.normalize import clean
from termmap.retrieval.cache import all_rows
from termmap.decision.coupling import effective_type
from termmap.reranking.prompts import pack_payload, TARGET_INSTRUCTIONS

PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class Cards:
    def __init__(self,cfg,profile="eval",terminology=None):
        out=term_dir(cfg);self.profile=profile;self.cfg=cfg
        self.icd=terminology["icd"] if terminology is not None else pd.read_parquet(out/("icd_concepts_final.parquet" if profile=="final" else "icd_concepts.parquet")).set_index("code").to_dict("index")
        self.sn=terminology["sn"] if terminology is not None else pd.read_parquet(out/"snomed_concepts.parquet").set_index("code").to_dict("index")
        self.held=set(pd.read_parquet(out/"probe_set.parquet").heldout_norm) if profile!="final" else set()
        aliases=terminology["aliases"] if terminology is not None else pd.read_parquet(out/("aliases_final.parquet" if profile=="final" else "aliases.parquet"))
        self.self_only_names=defaultdict(set)
        support=out/f"abbreviation_alias_support_{'final' if profile=='final' else 'train'}.parquet"
        if support.exists():
            for r in pd.read_parquet(support).itertuples():
                if len(r.support_groups)==1:self.self_only_names[(r.system,r.code,r.support_groups[0])].add(r.norm_text)
        self.aliases=defaultdict(list)
        for r in aliases.itertuples():
            if r.text not in self.aliases[(r.system,r.code)]:self.aliases[(r.system,r.code)].append(r.text)
        train=pd.read_parquet(out/"train.parquet") if profile!="final" else all_rows(cfg)
        self.examples=defaultdict(list)
        for r in train.loc[train.valid_gold].sort_values("row_id").itertuples():self.examples[(r.system,r.code)].append((r.entity,r.group_id))

    def card(self,row,code,seen=True):
        system=row["system"];node=(self.icd if system=="ICD" else self.sn)[code]
        excluded=self.held|({row["excluded_alias"]} if row.get("excluded_alias") else set())
        alias_excluded=excluded|self.self_only_names.get((system,code,row.get("excluded_group")),set())
        names=[t for t in self.aliases[(system,code)] if clean(strip_tag(t)) not in alias_excluded]
        preferred=node["preferred"] if clean(strip_tag(node["preferred"])) not in alias_excluded else (names[0] if names else "name withheld")
        if system=="ICD":
            parent=self.icd.get(node["parent"],{}).get("preferred","")
            category=self.icd[node["category"]]["preferred"]
            if clean(strip_tag(parent)) in excluded:parent="name withheld"
            if clean(strip_tag(category)) in excluded:category="name withheld"
            additions=node.get("refinement_text","")
            if clean(additions) in excluded:additions=""
            meaning=node.get("seventh_char_meaning","")
            if meaning and meaning not in additions:additions+="; "+meaning
            parts=[code,preferred,"adds over parent: "+additions,"parent: "+parent,"category: "+category,"leaf" if node["is_leaf"] else "header"]
            parts.append("also: "+"; ".join([t for t in names if t!=preferred][:2])[:400])
        else:
            fsn=node["fsn"] if clean(strip_tag(node["fsn"])) not in excluded else f"({node['tag']})"
            parts=[preferred,fsn,"top-level: "+node["top_level"],"also: "+"; ".join([t for t in names if t!=preferred][:3])[:400]]
            parts+= [f"{k}: {node[k]}" for k in ["laterality","acuity","absent","history","family_history"] if node.get(k)]
        if seen:
            examples=[t for t,g in self.examples[(system,code)] if g!=row.get("excluded_group") and clean(strip_tag(t)) not in excluded][:3]
            if examples:parts.append("Seen as: "+"; ".join(examples))
        return " | ".join(parts)

    def query(self,row,include_type=True):
        parts=[]
        if include_type and effective_type(row) not in {None,"Unknown",""}:parts.append("Type: "+effective_type(row))
        parts.append("Mention: "+row["entity"])
        views=row.get("views",{})
        enabled=self.cfg["retrieval"].get("enabled_by_system",{}).get(row["system"],self.cfg["retrieval"]["enabled_views"])
        if "V2" in enabled and "V2" in views and views["V2"]!=views.get("V1"):parts.append("Expansions: "+"; ".join(views["V2"]))
        possible=[t for v in ["V3","V4"] if v in enabled for t in views.get(v,[]) if t!=views.get("V1",[""])[0]]
        if possible:parts.append("Possible reading: "+"; ".join(dict.fromkeys(possible)))
        parts.append("Context: "+row.get("context",{}).get("window",f"[M]{row['entity']}[/M]"))
        return "\n".join(parts)

    def payload(self,row,code,include_type=True,seen=True):
        system="ICD-10-CM code" if row["system"]=="ICD" else "SNOMED CT concept"
        instruction=f"Given a clinical term and its chart context, decide whether the candidate {system} is the code a professional medical coder would assign."
        if self.cfg.get("reranker",{}).get("target_instruction",False):instruction=TARGET_INSTRUCTIONS[row["system"]]
        return f"<Instruct>: {instruction}\n<Query>: {self.query(row,include_type)}\n<Document>: {self.card(row,code,seen)}"


class Reranker:
    def __init__(self,cfg,adapter=None,train=False,profile="eval",terminology=None):
        from termmap.retrieval.dense import require_allocation
        require_allocation();seed_everything(13,True)
        base=json.loads((path_for(cfg,"external_dir")/"reranker_snapshot.json").read_text())["path"]
        base=str(Path(base) if Path(base).is_absolute() else cfg["root"]/base)
        self.tokenizer=AutoTokenizer.from_pretrained(base,padding_side="left",local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation="sdpa").cuda()
        self.model.config.use_cache=False
        if train:
            self.model=get_peft_model(self.model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules="all-linear",task_type="CAUSAL_LM"))
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            self.model.enable_input_require_grads()
        elif adapter:self.model=PeftModel.from_pretrained(self.model,str(adapter)).cuda()
        self.model.train(train)
        self.prefix=self.tokenizer.encode(PREFIX,add_special_tokens=False);self.suffix=self.tokenizer.encode(SUFFIX,add_special_tokens=False)
        self.yes=self.tokenizer.convert_tokens_to_ids("yes");self.no=self.tokenizer.convert_tokens_to_ids("no")
        self.cards=Cards(cfg,profile,terminology);self.cfg=cfg
        self.arch="sm"+"".join(str(x) for x in torch.cuda.get_device_capability())
        base_path=Path(base)
        weights=sorted(base_path.glob("*.safetensors")) or sorted(base_path.glob("pytorch_model*.bin"))
        base_hash=hashlib.sha256("".join(sha256(file) for file in weights).encode()).hexdigest()
        signature=dict(base_weights_sha256=base_hash,
                       base_config_sha256=sha256(base_path/"config.json"),
                       adapter_weights_sha256=sha256(Path(adapter)/"adapter_model.safetensors") if adapter else None,
                       adapter_config_sha256=sha256(Path(adapter)/"adapter_config.json") if adapter else None,
                       gpu_architecture=self.arch,implementation_sha256=sha256(Path(__file__)),
                       packing_sha256=sha256(Path(__file__).with_name("prompts.py")),
                       tokenizer_sha256={name:sha256(base_path/name) for name in ["tokenizer.json","tokenizer_config.json","special_tokens_map.json"] if (base_path/name).exists()},
                       prefix=PREFIX,suffix=SUFFIX,max_tokens=512)
        signature_hash=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        self.identity=("zero" if not adapter else signature["adapter_weights_sha256"][:20])+"_"+self.arch+"_"+signature_hash[:16]
        p=path_for(cfg,"run_dir")/"reranker_cache";p.mkdir(parents=True,exist_ok=True)
        write_json(p/(self.identity+".json"),signature)
        self.connection=sqlite3.connect(p/f"{self.identity}.sqlite")
        self.connection.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")

    def tokenize(self,payloads):
        payloads=[pack_payload(self.tokenizer,p,512-len(self.prefix)-len(self.suffix)) for p in payloads]
        encoded=self.tokenizer(payloads,padding=False,truncation=True,max_length=512-len(self.prefix)-len(self.suffix),add_special_tokens=False)["input_ids"]
        return self.tokenizer.pad({"input_ids":[self.prefix+x+self.suffix for x in encoded]},padding=True,return_tensors="pt").to("cuda")

    def logits(self,payloads):
        inputs=self.tokenize(payloads)
        base=self.model.get_base_model() if isinstance(self.model,PeftModel) else self.model
        hidden=base.model(**inputs,use_cache=False).last_hidden_state[:,-1,:]
        weights=base.lm_head.weight[[self.no,self.yes]]
        scores=torch.nn.functional.linear(hidden,weights).float()
        return scores[:,1]-scores[:,0]

    @torch.inference_mode()
    def score(self,payloads):
        # Cache the actual packed input. Unchanged fitting inputs retain their
        # existing keys; repaired inputs cannot reuse a candidate-blind score.
        payloads=[pack_payload(self.tokenizer,p,512-len(self.prefix)-len(self.suffix)) for p in payloads]
        result=[None]*len(payloads);missing=[]
        use_cache=self.cfg.get("reranker",{}).get("cache_enabled",True)
        for i,text in enumerate(payloads):
            key=hashlib.sha256(text.encode()).hexdigest();row=self.connection.execute("SELECT score FROM scores WHERE key=?",(key,)).fetchone() if use_cache else None
            if row is None:missing.append((i,key,text))
            else:result[i]=row[0]
        batch_size=self.cfg.get("reranker",{}).get("score_batch",64)
        for start in range(0,len(missing),batch_size):
            batch=missing[start:start+batch_size]
            scores=self.logits([t for _,_,t in batch]).cpu().numpy()
            rows=[]
            for (i,key,_),score in zip(batch,scores):result[i]=round(float(score),5);rows.append((key,result[i]))
            if use_cache:self.connection.executemany("INSERT OR REPLACE INTO scores VALUES (?,?)",rows);self.connection.commit()
        return result

    def score_items(self,items,seen=None):
        if seen is None:seen=self.cfg.get("reranker",{}).get("seen_enabled",True)
        self.model.eval();outputs=[]
        for i,item in enumerate(items):
            torch.cuda.synchronize();started=time.perf_counter()
            row=item["row"];codes=[c for c,_ in item["pool"]]
            payloads=[self.cards.payload(row,c,seen=seen) for c in codes]
            scores=self.score(payloads)
            output=dict(item);output["reranker_scores"]=dict(zip(codes,scores));outputs.append(output)
            if "other_system" in item:output["other_system"]=self.score_items([item["other_system"]],seen=seen)[0]
            torch.cuda.synchronize();output["rerank_seconds"]=time.perf_counter()-started
            if len(items)>50 and i%250==0:print("Reranker",self.identity,i,"/",len(items),flush=True)
        return outputs

    def close(self):
        self.connection.close();self.model.cpu();del self.model;gc.collect();torch.cuda.empty_cache()
