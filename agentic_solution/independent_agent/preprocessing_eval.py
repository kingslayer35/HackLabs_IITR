"""Query-view ablation on the frozen 500-input retrieval sample, under Slurm."""
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from collections import Counter

from benchmark_recall import sha,write,distinct_hits,SYSTEMS,KS,MODEL_ID,MAX_LENGTH
from retrieval_cascade_eval import bm25_search,cascade
from preprocessing import Preprocessor,context_diagnostics
from independent_agent.common import norm


def query_key(text):
    # Exact code lookup is case-sensitive in the original tools. Keep code
    # literals separate; other lexical operations depend on norm(text) only.
    if re.fullmatch(r'[A-Za-z]\d[\dA-Za-z.]*|\d+',text.strip()):return ('code',text.strip())
    return ('mention',norm(text))


def fuse_views(lists,limit=30):
    # Identical rankings vote once, e.g. lowercase views of an uncased model.
    lists=list(dict.fromkeys(tuple(x) for x in lists))
    scores={};original={c:i for i,c in enumerate(lists[0])}
    for cc in lists:
        for rank,code in enumerate(cc,1):scores[code]=scores.get(code,0)+1/(60+rank)
    return sorted(scores,key=lambda c:(-scores[c],original.get(c,100000),c))[:limit]


def main():
    assert os.environ.get('SLURM_JOB_ID')
    root=Path(__file__).resolve().parents[1];out=root/'results/preprocessing500_v2';out.mkdir(exist_ok=True)
    assert not (out/'results.json').exists(),'Do not overwrite completed runs'
    for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH'):
        assert Path(os.environ[key]).resolve().is_relative_to(root)
    import numpy as np
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')));torch.manual_seed(13);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    started=time.monotonic()
    data=root/'data/recall500_v1';old=root/'results/recall500_v1'
    xs=json.loads((data/'inputs.json').read_text());assert len(xs)==500
    cached=json.loads((old/'rankings.json').read_text());lexical=json.loads((old/'lexical_bm25_exact.json').read_text())
    previous=json.loads((root/'results/config.json').read_text())
    db_path=root/'data/terminology.sqlite';assert sha(db_path)==previous['source_sha256']['data/terminology.sqlite']
    sources=[data/'inputs.json',data/'labels.json',data/'selection_manifest.json',old/'rankings.json',old/'lexical_bm25_exact.json',db_path,Path(__file__),root/'code/preprocessing.py',root/'code/retrieval_cascade_eval.py',root/'code/benchmark_recall.py']
    hashes={str(p.relative_to(root)):sha(p) for p in sources}
    protocol=dict(input_count=500,source_sha256=hashes,model=MODEL_ID,model_revision=previous['model_revision'],
        model_parameters='unchanged CLS, max_length128, fp16 encode, normalized float32 exact search; cached aliases reused',
        base_backend=dict(ICD10CM='SapBERT top100',SNOMEDCT_US='fixed exact-first cascade: exact + four additional BM25 + dense to30'),
        query_view_fusion='equal-weight RRF k60 over distinct complete per-view rankings; original query breaks score ties; final top30',
        arms=dict(raw=['raw'],clean=['raw','clean'],abbreviation=['raw','clean','abbreviation'],repair=['raw','clean','repair'],combined=['raw','clean','abbreviation','repair']),
        abbreviations='dictionary ABBR - expansion / expansion (ABBR), require initials or single-word subsequence alignment, protect Roman numerals and ordinary words (non-all-uppercase dictionary occurrence>=5); up to3 expansions ordered by dictionary occurrence; pna/doe supplement; no label-derived priors',
        repair='only OOV alphabetic tokens len>=5; never known abbreviations/protected qualifiers/possessives; unique two-word split with each word freq>=5, otherwise unique missing-letter insertion, adjacent transposition or repeated-letter removal at vocab freq>=5; no arbitrary substitution/deletion',
        refinement='v2 rules tightened after manual review of v1 generated views on this development500; not an untouched held-out comparison',
        context='raw +/-300 character marked window and diagnostic cue flags saved only; not used for retrieval; no literal code injection',
        automatic_exact_selection=False,training_examples_used=False,training_performed=False,agent_or_LLM_API_calls=0,
        slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0))
    write(out/'config.json',protocol)
    db=sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)
    pp=Preprocessor(r[0] for r in db.execute('SELECT text FROM aliases ORDER BY id'))
    prepared={x['input_id']:dict(**pp.views(x['entity']),context=context_diagnostics(x['entity'],x['chunk_text'])) for x in xs}
    write(out/'query_views.json',prepared)
    write(out/'abbreviation_lexicon.json',{k:dict(v) for k,v in sorted(pp.abbreviations.items())})
    print('PREPARED',dict(Counter(v['kind'] for x in prepared.values() for v in x['views'])),flush=True)
    del pp
    query_strings=list(dict.fromkeys(v['text'] for x in prepared.values() for v in x['views']))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=previous['model_revision'],cache_dir=os.environ['HF_HOME'],local_files_only=True)
    def token_key(text):return tuple(tokenizer(text,truncation=True,max_length=MAX_LENGTH)['input_ids'])
    dense_cache={s:{} for s in SYSTEMS};bcache={};ecache={}
    for i,x in enumerate(xs):
        key=token_key(x['entity'])
        for s in SYSTEMS:
            cc=cached['sapbert'][s][x['input_id']]
            if key in dense_cache[s]:assert dense_cache[s][key]==cc
            dense_cache[s][key]=cc
        k=query_key(x['entity']);bcache[k]=lexical['bm25']['SNOMEDCT_US'][i];ecache[k]=lexical['exact']['SNOMEDCT_US'][i]
    tokens={q:token_key(q) for q in query_strings};new_keys=list(dict.fromkeys(k for k in tokens.values() if k not in dense_cache[SYSTEMS[0]]))
    print('NEW_DENSE_QUERIES',len(new_keys),'unique_query_strings',len(query_strings),flush=True)
    if new_keys:
        model=AutoModel.from_pretrained(MODEL_ID,revision=previous['model_revision'],cache_dir=os.environ['HF_HOME'],local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
        with torch.inference_mode():
            vectors=[]
            for begin in range(0,len(new_keys),64):
                encoded=tokenizer.pad([dict(input_ids=list(k),attention_mask=[1]*len(k)) for k in new_keys[begin:begin+64]],padding=True,return_tensors='pt')
                rep=model(**{k:v.to('cuda') for k,v in encoded.items()}).last_hidden_state[:,0,:]
                vectors.append(torch.nn.functional.normalize(rep.float(),p=2,dim=1))
            queries=torch.cat(vectors)
        for s in SYSTEMS:
            state=json.loads((root/'data/embeddings'/(s+'.state.json')).read_text())
            assert state['complete'] and state['config']['database_sha']==hashes['data/terminology.sqlite']
            assert state['config']['model_revision']==previous['model_revision'] and state['config']['max_length']==MAX_LENGTH and state['config']['order']=='len(text),alias_id'
            records=[(int(i),str(c),str(t)) for i,c,t in db.execute('SELECT id,code,text FROM aliases WHERE system=? ORDER BY id',(s,))];records.sort(key=lambda r:(len(r[2]),r[0]))
            embeddings=np.load(root/'data/embeddings'/(s+'.npy'),mmap_mode='r+')
            assert embeddings.shape[0]==len(records)==state['encoded_rows']
            with torch.inference_mode():
                emb_gpu=torch.as_tensor(np.asarray(embeddings),device='cuda')
                for begin in range(0,len(new_keys),32):
                    scores=queries[begin:begin+32]@emb_gpu.T
                    vals,indices=torch.topk(scores,k=min(1024,len(records)),dim=1,sorted=True);vals,indices=vals.cpu().numpy(),indices.cpu().numpy()
                    for j in range(len(vals)):
                        kk=len(vals[j]);found=distinct_hits(indices[j],vals[j],records,100)
                        while len(found)<100 and kk<len(records):
                            kk=min(kk*2,len(records));vv,ii=torch.topk(scores[j],k=kk,sorted=True);found=distinct_hits(ii.cpu().tolist(),vv.cpu().tolist(),records,100)
                        dense_cache[s][new_keys[begin+j]]=[r['code'] for r in found]
                    del scores
                del emb_gpu
            del records,embeddings
            torch.cuda.empty_cache();print('DENSE_DONE',s,flush=True)
    backend={s:{} for s in SYSTEMS};new_lexical=0
    for i,q in enumerate(query_strings):
        key=query_key(q)
        if key not in bcache:
            bcache[key]=bm25_search(db,'SNOMEDCT_US',q)
            direct=[r[0] for r in db.execute('SELECT code FROM concepts WHERE system=? AND code=?',('SNOMEDCT_US',q.strip()))]
            aliases=[r[0] for r in db.execute('SELECT DISTINCT code FROM aliases WHERE system=? AND normalized=? ORDER BY code',('SNOMEDCT_US',norm(q)))]
            ecache[key]=list(dict.fromkeys(direct+aliases));new_lexical+=1
        backend['ICD10CM'][q]=dense_cache['ICD10CM'][tokens[q]]
        backend['SNOMEDCT_US'][q]=cascade(ecache[key],bcache[key],dense_cache['SNOMEDCT_US'][tokens[q]])
        if (i+1)%100==0:print('VIEW_SEARCH',i+1,'/',len(query_strings),'new_bm25',new_lexical,flush=True)
    db.close()
    arms={}
    for arm,kinds in protocol['arms'].items():
        arms[arm]={s:{x['input_id']:fuse_views([backend[s][v['text']] for v in prepared[x['input_id']]['views'] if v['kind'] in kinds]) for x in xs} for s in SYSTEMS}
    for s,source in [('ICD10CM','sapbert'),('SNOMEDCT_US','cascade')]:
        assert all(arms['raw'][s][x['input_id']]==cached[source][s][x['input_id']][:30] for x in xs),'Raw control must exactly reproduce prior rankings'
    write(out/'rankings.json',arms)
    write(out/'per_query_rankings.json',backend)
    # Only load labels after all query generation and rankings are frozen.
    labels={r['input_id']:r['gold'] for r in json.loads((data/'labels.json').read_text())}
    metrics={};paired={}
    for arm,ss in arms.items():
        metrics[arm]={};paired[arm]={}
        for s,items in ss.items():
            metrics[arm][s]={};paired[arm][s]={}
            for k in KS:
                hit={i:labels[i][s] in cc[:k] for i,cc in items.items()};base={i:labels[i][s] in cc[:k] for i,cc in arms['raw'][s].items()}
                metrics[arm][s][str(k)]=dict(correct=sum(hit.values()),n=500,recall=sum(hit.values())/500)
                paired[arm][s][str(k)]=dict(recoveries=sum(hit[i] and not base[i] for i in labels),losses=sum(base[i] and not hit[i] for i in labels))
    result=dict(n=500,recall=metrics,paired_vs_raw=paired,changed_view_inputs={kind:sum(any(v['kind']==kind for v in p['views']) for p in prepared.values()) for kind in ['clean','abbreviation','repair']},
        new_dense_queries=len(new_keys),new_bm25_queries=new_lexical,verified_raw_control_matches_prior500=True,wall_seconds=round(time.monotonic()-started,3),agent_or_LLM_API_calls=0,
        note='Same development500; mention query-view preprocessing on fixed dense-ICD/cascade-SNOMED backends. Context diagnostics not used. Not agent accuracy.')
    assert all(sha(root/p)==v for p,v in hashes.items())
    write(out/'results.json',result);print('FINAL_RESULTS',json.dumps(result),flush=True)


if __name__=='__main__':main()
