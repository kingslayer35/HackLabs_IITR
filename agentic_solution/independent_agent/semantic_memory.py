"""Frozen CPU E5 case similarity over allowed training memory, with exact cosine search."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from .common import ARTIFACTS, norm, read_json
from .context_memory import ContextMemory, MAX_EXAMPLES, MAX_SYSTEM_BYTES, local_window
from .selector500_eval import sha, write

EXPERIMENT = ARTIFACTS / 'semantic_memory_v1'
MEMORY = ARTIFACTS / 'memory_context_v1'
MODEL_FILE = 'onnx/model_qint8_avx512_vnni.onnx'


def encoder_text(mention, context=None):
    # Symmetric case/paraphrase similarity: E5's documented query prefix on both sides.
    return 'query: ' + (mention if context is None else 'Mention: '+mention+'\nContext: '+context)


def average_pool_normalize(hidden, mask):
    mask = np.asarray(mask, dtype=np.float32)[..., None]
    pooled = (np.asarray(hidden,dtype=np.float32)*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1)
    return pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12)


class CPUEncoder:
    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model_dir/MODEL_FILE),sess_options=options,providers=['CPUExecutionProvider'])
        self.tokenizer=Tokenizer.from_file(str(model_dir/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=512);self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        assert self.session.get_outputs()[0].shape[-1]==384

    def encode(self,texts,batch_size=24):
        output=[];truncated=0
        for start in range(0,len(texts),batch_size):
            batch=self.tokenizer.encode_batch(texts[start:start+batch_size])
            truncated+=sum(bool(x.overflowing) for x in batch)
            inputs={
                'input_ids':np.asarray([x.ids for x in batch],dtype=np.int64),
                'attention_mask':np.asarray([x.attention_mask for x in batch],dtype=np.int64),
                'token_type_ids':np.asarray([x.type_ids for x in batch],dtype=np.int64)}
            hidden=self.session.run(None,inputs)[0]
            output.append(average_pool_normalize(hidden,inputs['attention_mask']))
            if start%480==0:print(f'Encoded {min(start+len(batch),len(texts))}/{len(texts)} unique texts',flush=True)
        return np.concatenate(output).astype(np.float32),truncated


def embed():
    if (EXPERIMENT/'embeddings.json').exists():
        for path,expected in read_json(EXPERIMENT/'embeddings.json')['hashes'].items():assert sha(Path(path))==expected,path
        print('Using verified frozen embeddings',flush=True);return
    files=read_json(EXPERIMENT/'model_files.json')
    for name,info in files['files'].items():assert sha(EXPERIMENT/'model'/name)==info['sha256']
    db=sqlite3.connect(f'file:{MEMORY}/memory.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
    rows=[dict(r) for r in db.execute('select * from cases order by id')];db.close()
    ids=read_json(MEMORY/'prepared.json')['input_ids']
    query_rows=[read_json(MEMORY/'evidence'/(iid+'.json'))['base']['payload'] for iid in ids]
    train_m=[encoder_text(r['entity']) for r in rows]
    train_c=[encoder_text(r['entity'],r['context']) for r in rows]
    query_m=[encoder_text(r['mention']) for r in query_rows]
    query_c=[encoder_text(r['mention'],local_window(r['mention'],r['context'])['text']) for r in query_rows]
    unique=list(dict.fromkeys(train_m+train_c+query_m+query_c));lookup={text:i for i,text in enumerate(unique)}
    write(EXPERIMENT/'embedding_inputs.json',dict(training_row_ids=[r['id'] for r in rows],query_input_ids=ids,
        unique_texts=unique,train_mention_indices=[lookup[t] for t in train_m],train_context_indices=[lookup[t] for t in train_c],
        query_mention_indices=[lookup[t] for t in query_m],query_context_indices=[lookup[t] for t in query_c]))
    started=time.monotonic();encoder=CPUEncoder(EXPERIMENT/'model');vectors,truncated=encoder.encode(unique)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    path=EXPERIMENT/'embeddings.npz'
    np.savez_compressed(path,train_mention=vectors[[lookup[t] for t in train_m]],train_context=vectors[[lookup[t] for t in train_c]],
        query_mention=vectors[[lookup[t] for t in query_m]],query_context=vectors[[lookup[t] for t in query_c]])
    sanity_texts=[encoder_text(x) for x in ['heart attack','myocardial infarction','kidney cyst','renal cyst','gait abnormality']]
    sanity,_=encoder.encode(sanity_texts);similarities=(sanity@sanity.T).tolist()
    checks=dict(heart_attack_to_infarction=similarities[0][1],heart_attack_to_kidney_cyst=similarities[0][2],
        kidney_to_renal_cyst=similarities[2][3],kidney_cyst_to_gait=similarities[2][4])
    assert checks['heart_attack_to_infarction']>checks['heart_attack_to_kidney_cyst']
    assert checks['kidney_to_renal_cyst']>checks['kidney_cyst_to_gait']
    protected=[path,EXPERIMENT/'embedding_inputs.json',MEMORY/'memory.sqlite',EXPERIMENT/'model_files.json']
    protected += [EXPERIMENT/'model'/name for name in files['files']]
    write(EXPERIMENT/'embeddings.json',dict(model=files['model'],revision=files['revision'],quantization='qint8_avx512_vnni',
        provider='CPUExecutionProvider',dimension=384,max_tokens=512,batch_size=24,threads=4,
        pooling='attention-mask mean pooling then L2 normalization',prefix='query: on both sides for symmetric case similarity',
        view_score='0.5 mention cosine + 0.5 mention-and-local-context cosine; no label/description in encoder text',
        training_rows=len(rows),query_inputs=len(ids),unique_texts=len(unique),truncated_unique_texts=truncated,
        seconds=time.monotonic()-started,versions={x:importlib.metadata.version(x) for x in ['onnxruntime','numpy','tokenizers']},
        sanity_checks=checks,hashes={str(p):sha(p) for p in protected}))
    print('EMBEDDINGS READY',checks,flush=True)


class SemanticMemory:
    def __init__(self,root=EXPERIMENT,memory_path=MEMORY/'memory.sqlite'):
        self.db=sqlite3.connect(f'file:{memory_path}?mode=ro',uri=True);self.db.row_factory=sqlite3.Row
        self.rows=[dict(r) for r in self.db.execute('select * from cases order by id')]
        archive=np.load(root/'embeddings.npz',allow_pickle=False)
        self.vectors={name:archive[name] for name in archive.files};archive.close()
        plan=read_json(root/'embedding_inputs.json');self.query_index={iid:i for i,iid in enumerate(plan['query_input_ids'])}
        assert [r['id'] for r in self.rows]==plan['training_row_ids']

    def close(self):self.db.close()

    def retrieve(self,iid,system,codes):
        query=self.query_index[iid];eligible=[i for i,r in enumerate(self.rows) if r['system']==system and r['code'] in codes]
        if not eligible:return dict(examples=[],selected_row_ids=[],ranked_row_ids=[],serialized_bytes=2,eligible_rows=0)
        scores=.5*(self.vectors['train_mention'][eligible]@self.vectors['query_mention'][query])
        scores+=.5*(self.vectors['train_context'][eligible]@self.vectors['query_context'][query])
        ordered=sorted(zip(eligible,scores),key=lambda v:(-float(v[1]),self.rows[v[0]]['id']))
        examples=[];seen=set();used=2;selected=[]
        for index,score in ordered:
            row=self.rows[index];value=ContextMemory._format(row)
            if value is None or norm(row['entity']) in seen:continue
            size=len(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode())+int(bool(examples))
            if used+size>MAX_SYSTEM_BYTES:continue
            assert row['code'] in codes
            examples.append(value);selected.append(row['id']);seen.add(norm(row['entity']));used+=size
            if len(examples)==MAX_EXAMPLES:break
        return dict(examples=examples,selected_row_ids=selected,ranked_row_ids=[self.rows[i]['id'] for i,_ in ordered],
            selected_cosines=[float(score) for i,score in ordered if self.rows[i]['id'] in selected],serialized_bytes=used,eligible_rows=len(eligible))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['embed']);args=parser.parse_args();embed()
