"""Flat dictionary tools backed by Slurm SapBERT search and hidden result pools."""
import copy
import requests

from .common import digest
from .comparisons import PolicyTools,schemas_for,assert_flat
from .tools import tool_schema

MORE_SCHEMA=tool_schema('get_more_candidates',
    'Read a deeper page from a previous search without repeating it. Use search_id and next_offset from search_concepts or this tool when the visible candidates do not fit. Hidden candidates have not yet been inspected. Each page returns at most12 codes.',
    dict(search_id=dict(type='string'),offset=dict(type='integer',minimum=0,maximum=99),limit=dict(type='integer',minimum=1,maximum=12)),['search_id','offset'])


def agent_schemas(final_turn=False):
    schemas=schemas_for('search_inspect',final_turn=final_turn)
    if final_turn:return schemas
    for s in schemas:
        if s['function']['name']=='search_concepts':
            s['function']['description']='Search supplied dictionaries using SapBERT for ICD and exact/BM25/SapBERT cascade for SNOMED. Reformulate abbreviations or clinical wording if needed. Returns a page plus search_id and next_offset for deeper results using get_more_candidates. Rank is not clinical confidence.'
    return copy.deepcopy(schemas[:-1]+[MORE_SCHEMA]+schemas[-1:])


class DenseAgentTools(PolicyTools):
    def __init__(self,example_path,url='http://127.0.0.1:18473',fetch=None):
        super().__init__(example_path,graph=False)
        self.url=url;self.fetch=fetch;self.pools={};self.exposed={s:set() for s in self.seen};self.pages_read=[]

    def _fetch(self,system,queries):
        if self.fetch:return self.fetch(system,queries)
        try:
            response=requests.post(self.url+'/search',json=dict(system=system,queries=queries),timeout=(5,120));response.raise_for_status()
            return response.json()
        except (requests.RequestException,ValueError) as exc:
            raise RuntimeError('GPU retrieval service failed: '+type(exc).__name__) from None

    def _page(self,search_id,offset,limit):
        pool=self.pools[search_id];system=pool['system'];rows=pool['results'][offset:offset+limit]
        result=[]
        for rank,row in enumerate(rows,offset+1):
            card=self.card(system,row['code'])
            if card is None:raise RuntimeError('Search service returned unknown dictionary code')
            card.update(matched_alias=row.get('matched_alias',''),retrieval_score=row.get('retrieval_score'),rank=rank)
            result.append(card);self.exposed[system].add(row['code'])
        next_offset=offset+len(rows)
        payload=dict(system=system,queries=pool['queries'],search_id=search_id,offset=offset,results=result,
            total_candidates=len(pool['results']),next_offset=next_offset if next_offset<len(pool['results']) else None,
            has_more=next_offset<len(pool['results']),note='Rank is retrieval evidence, not clinical confidence. Request another page or reformulate if meaning is absent.')
        assert_flat(payload)
        return payload

    def search_concepts(self,system,queries,limit=12):
        self._system(system)
        if isinstance(queries,str):queries=[queries]
        if not isinstance(queries,list) or not 1<=len(queries)<=6:raise ValueError('Provide 1 to6 queries')
        if any(not isinstance(q,str) or len(q)>500 for q in queries):raise ValueError('Each query must be a string <=500 characters')
        limit=min(max(int(limit),1),30)
        search_id=digest(system+'\0'+'\0'.join(queries))[:20]
        if search_id not in self.pools:
            data=self._fetch(system,queries)
            if len(data['results'])>100 or len({r['code'] for r in data['results']})!=len(data['results']):raise RuntimeError('Invalid retrieval pool')
            self.pools[search_id]=dict(system=system,queries=queries,results=data['results'])
        payload=self._page(search_id,0,limit)
        self.search_log.append(dict(system=system,queries=queries,search_id=search_id,codes=[r['code'] for r in payload['results']]))
        return payload

    def get_more_candidates(self,search_id,offset,limit=12):
        if search_id not in self.pools:raise ValueError('Unknown search_id; use a prior search from this input')
        if not isinstance(offset,int) or isinstance(offset,bool) or not 0<=offset<len(self.pools[search_id]['results']):raise ValueError('Offset outside previous search pool')
        if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=12:raise ValueError('Page limit must be between1 and12')
        payload=self._page(search_id,offset,limit)
        self.pages_read.append(dict(search_id=search_id,offset=offset,codes=[r['code'] for r in payload['results']]))
        return payload

    def dispatch(self,name,arguments):
        if name=='get_more_candidates':return self.get_more_candidates(**arguments)
        return super().dispatch(name,arguments)
