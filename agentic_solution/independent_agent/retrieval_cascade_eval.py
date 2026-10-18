"""Fixed exact/BM25/dense cascade evaluation; run only under Slurm.

Remote companion benchmark_recall.py is the frozen 80-input retrieval benchmark.
No demonstrations, context queries, agent or LLM calls are used.
"""
import csv
import json
import os
from pathlib import Path
import sqlite3
import time

from benchmark_recall import sha, write, distinct_hits, fuse, SYSTEMS, KS, MODEL_ID, MAX_LENGTH
from independent_agent.common import norm
from independent_agent.tools import TerminologyTools, query_terms, fts_query


def unique(codes):
    return list(dict.fromkeys(codes))


def cascade(exact, bm25, dense, children=(), limit=30):
    # Fixed quotas, independent of labels or system. An exact match is priority,
    # not an automatic final prediction. Up to four *additional* BM25 codes.
    result = unique([*exact, *children])
    added = 0
    for code in bm25:
        if code not in result:
            result.append(code)
            added += 1
            if added == 4:
                break
    return unique([*result, *dense, *bm25])[:limit]


def bm25_search(db, system, entity):
    terms = query_terms(entity)
    if not terms:
        return []
    rows = db.execute('''SELECT a.code,bm25(alias_fts) AS relevance
        FROM alias_fts JOIN aliases a ON a.id=alias_fts.rowid
        WHERE alias_fts MATCH ? AND a.system=?
        ORDER BY relevance,a.code,a.id LIMIT 350''', (fts_query(terms, 'OR'), system))
    return unique([str(r[0]) for r in rows])[:100]


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run GPU operations through Slurm'
    root = Path(__file__).resolve().parents[1]
    data = root/'data/recall500_v1'
    out = root/'results/recall500_v1'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'results.json').exists(), 'Do not overwrite completed evaluation'
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.manual_seed(13)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    for key in ('TMPDIR','HF_HOME','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH'):
        assert Path(os.environ[key]).resolve().is_relative_to(root)
    started = time.monotonic()
    inputs = json.loads((data/'inputs.json').read_text())
    labels = {r['input_id']:r['gold'] for r in json.loads((data/'labels.json').read_text())}
    assert len(inputs) == 500 and len(labels) == 500
    assert all(set(r) == {'input_id','entity','chunk_text'} for r in inputs)
    old_inputs = json.loads((root/'data/inputs.json').read_text())
    assert not ({r['input_id'] for r in inputs} & {r['input_id'] for r in old_inputs})
    original = json.loads((root/'results/config.json').read_text())
    db_path = root/'data/terminology.sqlite'
    assert sha(db_path) == original['source_sha256']['data/terminology.sqlite']
    hashes = {str(p.relative_to(root)):sha(p) for p in (data/'inputs.json',data/'labels.json',data/'selection_manifest.json',db_path,Path(__file__),root/'code/benchmark_recall.py',root/'code/independent_agent/tools.py')}
    protocol = dict(n=500, source_sha256=hashes, model=MODEL_ID, model_revision=original['model_revision'],
        max_length=MAX_LENGTH, pooling='CLS before pooler; L2 float32', inference_dtype='float16', search='exact float32; TF32 disabled; max alias score per code',
        dense_pool=100, bm25_alias_pool=350, bm25_query='OR over existing stopword-filtered first 14 distinct normalized terms; existing Porter FTS index',
        cascade='exact code then normalized exact aliases, then up to 4 additional BM25 codes, then dense, then remaining BM25; deduplicate and truncate',
        category_cascade='after exact matches insert up to 2 direct children of non-leaf exact matches present in dense top100, in dense rank order; then same cascade',
        category_source='supplied derived ICD prefix hierarchy; no official SNOMED hierarchy available',
        rrf='same equal-weight RRF k=60, lexical top30 and dense top100',
        automatic_exact_selection=False, fixed_before_scoring=True, training_examples_used=False, context_used=False,
        agent_or_LLM_API_calls=0, slurm_job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),ks=list(KS))
    write(out/'config.json',protocol)
    print('CONFIG',json.dumps(protocol),flush=True)
    db = sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)
    tools = TerminologyTools(db_path)
    lexical={s:[] for s in SYSTEMS}; bm25={s:[] for s in SYSTEMS}; exact={s:[] for s in SYSTEMS}
    for i,item in enumerate(inputs):
        for s in SYSTEMS:
            lexical[s].append([r['code'] for r in tools.search_concepts(s,[item['entity']],30)['results']])
            bm25[s].append(bm25_search(db,s,item['entity']))
            direct=[r[0] for r in db.execute('SELECT code FROM concepts WHERE system=? AND code=?',(s,item['entity'].strip()))]
            aliases=[r[0] for r in db.execute('SELECT DISTINCT code FROM aliases WHERE system=? AND normalized=? ORDER BY code',(s,norm(item['entity'])))]
            exact[s].append(unique(direct+aliases))
        if (i+1)%50==0:print('LEXICAL_AND_BM25',i+1,'/500',flush=True)
    tools.close()
    write(out/'lexical_bm25_exact.json',dict(lexical=lexical,bm25=bm25,exact=exact))
    model=AutoModel.from_pretrained(MODEL_ID,revision=original['model_revision'],cache_dir=os.environ['HF_HOME'],local_files_only=True,torch_dtype=torch.float16).to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=original['model_revision'],cache_dir=os.environ['HF_HOME'],local_files_only=True)
    with torch.inference_mode():
        queries=[]
        for begin in range(0,len(inputs),64):
            toks=tokenizer([r['entity'] for r in inputs[begin:begin+64]],padding=True,truncation=True,max_length=MAX_LENGTH,return_tensors='pt')
            v=model(**{k:v.to('cuda') for k,v in toks.items()}).last_hidden_state[:,0,:]
            queries.append(torch.nn.functional.normalize(v.float(),p=2,dim=1))
        queries=torch.cat(queries)
    dense={}; details={}; children={s:[] for s in SYSTEMS}
    for s in SYSTEMS:
        state=json.loads((root/'data/embeddings'/(s+'.state.json')).read_text())
        assert state['complete'] and state['config']['database_sha']==hashes['data/terminology.sqlite']
        assert state['config']['model_revision']==original['model_revision'] and state['config']['max_length']==MAX_LENGTH
        assert state['config']['order']=='len(text),alias_id' and state['config']['inference_dtype']=='float16'
        records=[(int(i),str(c),str(t)) for i,c,t in db.execute('SELECT id,code,text FROM aliases WHERE system=? ORDER BY id',(s,))]
        records.sort(key=lambda r:(len(r[2]),r[0]))
        vectors=np.load(root/'data/embeddings'/(s+'.npy'),mmap_mode='r+')
        assert len(records)==state['encoded_rows']==vectors.shape[0]
        with torch.inference_mode():
            v_gpu=torch.as_tensor(np.asarray(vectors),device='cuda')
            rr=[];dd=[]
            for begin in range(0,len(inputs),32):
                scores=queries[begin:begin+32]@v_gpu.T
                vals,indices=torch.topk(scores,k=min(1024,len(records)),dim=1,sorted=True)
                vals,indices=vals.cpu().numpy(),indices.cpu().numpy()
                for j in range(len(vals)):
                    kk=len(vals[j]); found=distinct_hits(indices[j],vals[j],records,100)
                    while len(found)<100 and kk<len(records):
                        kk=min(kk*2,len(records));vv,ii=torch.topk(scores[j],k=kk,sorted=True)
                        found=distinct_hits(ii.cpu().tolist(),vv.cpu().tolist(),records,100)
                    rr.append([r['code'] for r in found]);dd.append(found)
                del scores
            del v_gpu
        dense[s]=rr;details[s]=dd
        for i,codes in enumerate(exact[s]):
            eligible=set()
            for code in codes:
                node=db.execute('SELECT is_leaf FROM concepts WHERE system=? AND code=?',(s,code)).fetchone()
                if node and not node[0]:
                    eligible.update(r[0] for r in db.execute('SELECT code FROM concepts WHERE system=? AND parent=?',(s,code)))
            children[s].append([c for c in rr[i] if c in eligible and c not in codes][:2])
        del vectors,records
        torch.cuda.empty_cache()
        print('DENSE_DONE',s,flush=True)
    db.close()
    rankings=dict(lexical=lexical,bm25=bm25,sapbert=dense)
    rankings['rrf']={s:[fuse(l,d,30) for l,d in zip(lexical[s],dense[s])] for s in SYSTEMS}
    rankings['cascade']={s:[cascade(e,b,d) for e,b,d in zip(exact[s],bm25[s],dense[s])] for s in SYSTEMS}
    rankings['category_cascade']={s:[cascade(e,b,d,c) for e,b,d,c in zip(exact[s],bm25[s],dense[s],children[s])] for s in SYSTEMS}
    # Persist all rankings before label-based scoring.
    write(out/'rankings.json',{a:{s:{x['input_id']:cc for x,cc in zip(inputs,ss[s])} for s in SYSTEMS} for a,ss in rankings.items()})
    write(out/'dense_alias_details.json',{s:{x['input_id']:dd for x,dd in zip(inputs,details[s])} for s in SYSTEMS})
    write(out/'category_expansions.json',{s:{x['input_id']:c for x,c in zip(inputs,children[s]) if c} for s in SYSTEMS})
    summary={}; paired={}; rows=[]; coverage={}; exact_audit={}
    check=sqlite3.connect(f'file:{db_path}?mode=ro',uri=True)
    for s in SYSTEMS:
        coverage[s]=sum(check.execute('SELECT 1 FROM concepts WHERE system=? AND code=?',(s,labels[x['input_id']][s])).fetchone() is not None for x in inputs)
        exact_audit[s]=dict(no_exact=sum(not e for e in exact[s]),unique=sum(len(e)==1 for e in exact[s]),
            unique_gold_agreement=sum(len(e)==1 and labels[x['input_id']][s] in e for x,e in zip(inputs,exact[s])),
            ambiguous=sum(len(e)>1 for e in exact[s]),category_expanded_inputs=sum(bool(c) for c in children[s]))
    check.close()
    for arm,ss in rankings.items():
        summary[arm]={};paired[arm]={}
        for s in SYSTEMS:
            summary[arm][s]={};paired[arm][s]={}
            for k in KS:
                hit=[labels[x['input_id']][s] in cc[:k] for x,cc in zip(inputs,ss[s])]
                base=[labels[x['input_id']][s] in cc[:k] for x,cc in zip(inputs,lexical[s])]
                rrf_hit=[labels[x['input_id']][s] in cc[:k] for x,cc in zip(inputs,rankings['rrf'][s])]
                summary[arm][s][str(k)]=dict(correct=sum(hit),n=500,recall=sum(hit)/500)
                paired[arm][s][str(k)]=dict(vs_lexical=dict(recoveries=sum(a and not b for a,b in zip(hit,base)),losses=sum(b and not a for a,b in zip(hit,base))),vs_rrf=dict(recoveries=sum(a and not b for a,b in zip(hit,rrf_hit)),losses=sum(b and not a for a,b in zip(hit,rrf_hit))))
            for x,cc in zip(inputs,ss[s]):
                assert len(cc)==len(set(cc))
                g=labels[x['input_id']][s]
                rows.append(dict(arm=arm,system=s,input_id=x['input_id'],entity=x['entity'],gold=g,gold_rank=cc.index(g)+1 if g in cc else None,**{f'hit_at_{k}':g in cc[:k] for k in KS}))
    result=dict(n=500,recall=summary,paired=paired,dictionary_gold_coverage=coverage,exact_audit=exact_audit,wall_seconds=round(time.monotonic()-started,3),agent_or_LLM_API_calls=0,note='500 seeded dev/test inputs excluding prior80 IDs; dictionary mention-only recall, not agent accuracy. Unavailable gold remains in denominator.')
    assert all(sha(root/p)==v for p,v in hashes.items())
    write(out/'results.json',result)
    with (out/'per_input_recall.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('FINAL_RESULTS',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
