"""SapBERT CLS encoding and exact chunked torch inner products, Slurm only."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer
from termmap.retrieval.lexical import rank_codes


def require_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU work requires a Slurm allocation")
    if not torch.cuda.is_available():raise RuntimeError("Allocated GPU is unavailable")


class Encoder:
    def __init__(self,model_path,max_length=40,batch_size=256):
        require_allocation()
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        self.model=AutoModel.from_pretrained(str(model_path),local_files_only=True).cuda().eval()
        self.max_length=max_length;self.batch_size=batch_size

    @torch.inference_mode()
    def encode(self,texts,target:Path|None=None):
        if target is None:result=np.empty((len(texts),self.model.config.hidden_size),dtype=np.float16)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+".partial")
            result=np.lib.format.open_memmap(temporary,mode="w+",dtype=np.float16,shape=(len(texts),self.model.config.hidden_size))
        for start in range(0,len(texts),self.batch_size):
            tokens=self.tokenizer(texts[start:start+self.batch_size],padding="max_length",truncation=True,
                                  max_length=self.max_length,return_tensors="pt").to("cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):output=self.model(**tokens).last_hidden_state[:,0,:]
            result[start:start+len(output)]=torch.nn.functional.normalize(output.float(),dim=-1).cpu().numpy().astype(np.float16)
            if target is not None and start%(self.batch_size*100)==0:print(f"Encoded {start}/{len(texts)}",flush=True)
        if hasattr(result,"flush"):result.flush()
        if target is not None:
            del result;temporary.replace(target)
            return np.load(target,mmap_mode="r")
        return result


class DenseIndex:
    def __init__(self,aliases,embeddings,chunk=32768):
        require_allocation()
        self.aliases=aliases.reset_index(drop=True)
        self.codes=np.array(sorted(aliases.code.unique()),dtype=object)
        lookup={c:i for i,c in enumerate(self.codes)}
        self.alias_codes=torch.tensor([lookup[c] for c in aliases.code],dtype=torch.int64,device="cuda")
        self.embeddings=torch.tensor(np.asarray(embeddings),device="cuda",dtype=torch.float16)
        self.chunk=chunk

    @torch.inference_mode()
    def score_all(self,queries,allowed=None,exclude_group=None):
        q=torch.tensor(np.asarray(queries),device="cuda",dtype=torch.float16)
        maxima=torch.full((len(q),len(self.codes)),-1.0,device="cuda",dtype=torch.float32)
        if allowed is None:allowed=np.ones(len(self.aliases),dtype=bool)
        if exclude_group is not None and "group_id" in self.aliases:
            allowed=allowed & self.aliases.group_id.ne(exclude_group).to_numpy()
        for start in range(0,len(self.embeddings),self.chunk):
            end=min(start+self.chunk,len(self.embeddings))
            values=(q @ self.embeddings[start:end].T).float()
            values=torch.round(values,decimals=5)
            mask=torch.tensor(allowed[start:end],device="cuda")
            values[:,~mask]=-1.0
            idx=self.alias_codes[start:end].expand(len(q),-1)
            maxima.scatter_reduce_(1,idx,values,reduce="amax",include_self=True)
        return maxima.cpu().numpy()

    def search(self,queries,k=100,allowed=None,exclude_group=None):
        return [rank_codes(row,self.codes,k) for row in self.score_all(queries,allowed,exclude_group)]
