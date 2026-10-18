"""Loopback-only terminology search service; GPU execution requires Slurm.

No agent, API credentials, examples or evaluation labels are loaded here.
"""
import json
import os
from pathlib import Path
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from benchmark_recall import MODEL_ID,MAX_LENGTH,SYSTEMS,sha,write,distinct_hits
from retrieval_cascade_eval import bm25_search,cascade
from independent_agent.common import norm,digest


class Engine:
    def __init__(self):
        assert os.environ.get('SLURM_JOB_ID'),'GPU service must run inside Slurm'
        import torch
        import numpy as np
        from transformers import AutoModel,AutoTokenizer
        self.torch=torch;self.np=np
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
        torch.manual_seed(13);torch.backends.cuda.matmul.allow_tf32=False
        self.root=Path(__file__).resolve().parents[1]
        for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH'):
            assert Path(os.environ[key]).resolve().is_relative_to(self.root)
        self.out=self.root/'results/agent_retrieval_v1';self.out.mkdir(exist_ok=True)
        dbpath=self.root/'data/terminology.sqlite'
        old=json.loads((self.root/'results/config.json').read_text())
        database_sha=sha(dbpath);assert database_sha==old['source_sha256']['data/terminology.sqlite']
        self.db=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
        self.model=AutoModel.from_pretrained(MODEL_ID,revision=old['model_revision'],cache_dir=os.environ['HF_HOME'],local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=old['model_revision'],cache_dir=os.environ['HF_HOME'],local_files_only=True)
        self.records={};self.vectors={};self.dense_cache={};self.query_cache={}
        for s in SYSTEMS:
            state=json.loads((self.root/'data/embeddings'/(s+'.state.json')).read_text())
            assert state['complete'] and state['config']['database_sha']==database_sha
            assert state['config']['model_revision']==old['model_revision'] and state['config']['max_length']==MAX_LENGTH and state['config']['order']=='len(text),alias_id'
            rr=[(int(i),str(c),str(t)) for i,c,t in self.db.execute('SELECT id,code,text FROM aliases WHERE system=? ORDER BY id',(s,))]
            rr.sort(key=lambda r:(len(r[2]),r[0]));self.records[s]=rr
            arr=np.load(self.root/'data/embeddings'/(s+'.npy'),mmap_mode='r+')
            assert len(arr)==len(rr)==state['encoded_rows']
            self.vectors[s]=torch.as_tensor(np.asarray(arr),device='cuda')
        # Preserve the already-computed raw80 dense rankings exactly.
        raw=json.loads((self.root/'data/inputs.json').read_text())
        for s in SYSTEMS:
            details=json.loads((self.root/'results'/(s+'_dense_rankings.json')).read_text())
            for x in raw:self.dense_cache[(s,x['entity'])]=details[x['input_id']]
        self.manifest=dict(model=MODEL_ID,model_revision=old['model_revision'],dictionary_sha256=database_sha,
            source_sha256={str(p.relative_to(self.root)):sha(p) for p in (Path(__file__),self.root/'code/benchmark_recall.py',self.root/'code/retrieval_cascade_eval.py')},
            backend=dict(ICD10CM='SapBERT max-alias-score exact search top100',SNOMEDCT_US='exact + four additional BM25 + dense to100'),
            multiquery='RRF constant20, ranks starting0, same as original lexical tool',
            max_pool=100,max_length=MAX_LENGTH,gpu=torch.cuda.get_device_name(0),slurm_job_id=os.environ['SLURM_JOB_ID'],
            labels_loaded=False,training_examples_loaded=False,preprocessing_enabled=False)
        write(self.out/'service_config.json',self.manifest)
        print('SERVICE_READY',json.dumps(self.manifest),flush=True)

    def dense(self,system,query):
        key=(system,query)
        if key in self.dense_cache:return self.dense_cache[key]
        torch=self.torch
        with torch.inference_mode():
            toks=self.tokenizer(query,truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
            rep=self.model(**{k:v.to('cuda') for k,v in toks.items()}).last_hidden_state[:,0,:]
            rep=torch.nn.functional.normalize(rep.float(),p=2,dim=1)
            scores=(rep@self.vectors[system].T)[0]
            kk=min(1024,len(scores));vals,ids=torch.topk(scores,k=kk,sorted=True)
            found=distinct_hits(ids.cpu().tolist(),vals.cpu().tolist(),self.records[system],100)
            while len(found)<100 and kk<len(scores):
                kk=min(kk*2,len(scores));vals,ids=torch.topk(scores,k=kk,sorted=True)
                found=distinct_hits(ids.cpu().tolist(),vals.cpu().tolist(),self.records[system],100)
        self.dense_cache[key]=found
        return found

    def one(self,system,query):
        key=(system,query)
        if key in self.query_cache:return self.query_cache[key]
        dense=self.dense(system,query);codes=[r['code'] for r in dense]
        aliases={r['code']:r['matched_alias'] for r in dense}
        if system=='SNOMEDCT_US':
            direct=[r[0] for r in self.db.execute('SELECT code FROM concepts WHERE system=? AND code=?',(system,query.strip()))]
            exact=[r[0] for r in self.db.execute('SELECT DISTINCT code FROM aliases WHERE system=? AND normalized=? ORDER BY code',(system,norm(query)))]
            codes=cascade(list(dict.fromkeys(direct+exact)),bm25_search(self.db,system,query),codes,limit=100)
        result=[dict(code=c,matched_alias=aliases.get(c,'')) for c in codes]
        self.query_cache[key]=result
        write(self.out/'query_cache'/(digest(system+'\0'+query)+'.json'),dict(system=system,query=query,results=result))
        return result

    def search(self,system,queries):
        if system not in SYSTEMS:raise ValueError('Unknown system')
        if not isinstance(queries,list) or not 1<=len(queries)<=6:raise ValueError('Provide 1 to 6 queries')
        if any(not isinstance(q,str) or len(q)>500 for q in queries):raise ValueError('Each query must be a string <=500 characters')
        scores={};aliases={};matched={}
        for q in queries:
            for rank,row in enumerate(self.one(system,q)):
                c=row['code'];scores[c]=scores.get(c,0)+1/(20+rank)
                aliases.setdefault(c,row['matched_alias']);matched.setdefault(c,[]).append(q)
        codes=sorted(scores,key=lambda c:(-scores[c],c))[:100]
        return dict(system=system,queries=queries,results=[dict(code=c,matched_alias=aliases[c],retrieval_score=round(scores[c],7)) for c in codes])


def main():
    engine=Engine()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,value):
            data=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path!='/health':return self.send(404,dict(error='Not found'))
            self.send(200,engine.manifest)
        def do_POST(self):
            if self.path!='/search':return self.send(404,dict(error='Not found'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=20000:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size));started=time.monotonic()
                result=engine.search(payload['system'],payload['queries']);result['seconds']=round(time.monotonic()-started,3)
                self.send(200,result)
            except (ValueError,KeyError,TypeError) as exc:self.send(400,dict(error=str(exc)))
        def log_message(self,*args):pass
    HTTPServer(('127.0.0.1',18473),Handler).serve_forever()


if __name__=='__main__':main()
