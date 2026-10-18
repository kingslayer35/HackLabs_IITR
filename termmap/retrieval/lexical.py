"""Deterministic lexical channels, exact retrieval, RRF and bounded expansion."""
from __future__ import annotations
import time
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def rank_codes(scores, codes, k):
    scores=np.round(np.asarray(scores,dtype=np.float64),5)
    if len(scores)>k:
        threshold=np.partition(scores,len(scores)-k)[len(scores)-k]
        ids=np.flatnonzero(scores>=threshold)
    else: ids=np.arange(len(scores))
    return sorted(((str(codes[i]),float(scores[i])) for i in ids),key=lambda x:(-x[1],x[0]))[:k]


class CharIndex:
    def __init__(self, aliases):
        dedup=["code","norm_text"]+(["group_id"] if "group_id" in aliases else [])
        self.aliases=aliases.drop_duplicates(dedup).reset_index(drop=True)
        self.codes=np.array(sorted(self.aliases.code.unique()),dtype=object)
        code_index={c:i for i,c in enumerate(self.codes)}
        self.alias_codes=np.array([code_index[c] for c in self.aliases.code],dtype=np.int32)
        self.vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,4),dtype=np.float32)
        self.matrix=self.vectorizer.fit_transform(self.aliases.text)

    def search(self, texts, k=100, exclude_group=None):
        query=self.vectorizer.transform(texts)
        similarities=query @ self.matrix.T
        rows=[]
        for i in range(len(texts)):
            row=similarities.getrow(i)
            scores=np.zeros(len(self.codes),dtype=np.float32)
            keep=np.ones(len(row.indices),dtype=bool)
            if exclude_group is not None and "group_id" in self.aliases:
                keep=self.aliases.group_id.to_numpy()[row.indices]!=exclude_group
            np.maximum.at(scores,self.alias_codes[row.indices[keep]],row.data[keep])
            rows.append(rank_codes(scores,self.codes,k))
        return rows

    def score_all(self,texts,exclude_norm=None):
        similarities=self.vectorizer.transform(texts)@self.matrix.T
        result=np.zeros((len(texts),len(self.codes)),dtype=np.float32)
        for i in range(len(texts)):
            row=similarities.getrow(i)
            keep=np.ones(len(row.indices),dtype=bool) if exclude_norm is None else self.aliases.norm_text.to_numpy()[row.indices]!=exclude_norm
            np.maximum.at(result[i],self.alias_codes[row.indices[keep]],row.data[keep])
        return result

    def scores_without_alias(self,texts,exclude_norm,base_scores):
        """Only concepts containing a removed name can change their max score."""
        if not hasattr(self,"code_alias_indices"):
            self.code_alias_indices=self.aliases.groupby("code",sort=False).indices
            self.norm_codes=self.aliases.groupby("norm_text",sort=False).code.agg(lambda x:tuple(set(x))).to_dict()
            self.code_positions={c:i for i,c in enumerate(self.codes)}
        result=np.asarray(base_scores,dtype=np.float32).copy();query=self.vectorizer.transform(texts)
        for code in self.norm_codes.get(exclude_norm,()):
            ids=self.code_alias_indices[code];ids=ids[self.aliases.norm_text.to_numpy()[ids]!=exclude_norm]
            value=float((query@self.matrix[ids].T).max()) if len(ids) else 0.
            result[self.code_positions[code]]=value
        return result

    def scores_without_pairs(self,texts,excluded_pairs,base_scores,exclude_norm=None):
        """Mask learned code/name assignments while retaining other code senses."""
        result=np.asarray(base_scores,dtype=np.float32).copy()
        query=self.vectorizer.transform(texts)
        for code in sorted({c for c,_ in excluded_pairs}):
            mask=self.aliases.code.eq(code)
            mask &= ~self.aliases.norm_text.isin([n for c,n in excluded_pairs if c==code])
            if exclude_norm:mask &= self.aliases.norm_text.ne(exclude_norm)
            ids=np.flatnonzero(mask.to_numpy())
            value=float((query@self.matrix[ids].T).max()) if len(ids) else 0.
            result[np.flatnonzero(self.codes==code)[0]]=value
        return result


class ConceptCharIndex(CharIndex):
    """C0 represents each dictionary concept by all its names in one document.

    Stage 2 still uses max-over-alias channels; this class is the independent
    lexical baseline, whose dictionary document granularity must be explicit.
    """
    def __init__(self,aliases):
        corpus=aliases.groupby("code",sort=True).text.agg(lambda x:" ".join(dict.fromkeys(x))).reset_index()
        corpus["norm_text"]=corpus.text
        super().__init__(corpus)


class BM25Index:
    def __init__(self, aliases):
        import bm25s
        self.aliases=aliases.drop_duplicates(["code","norm_text"]).reset_index(drop=True)
        self.codes=self.aliases.code.to_numpy()
        corpus=bm25s.tokenize(self.aliases.text.tolist(),stopwords="en",show_progress=False)
        self.engine=bm25s.BM25()
        self.engine.index(corpus,show_progress=False)

    def search(self,texts,k=100,exclude_norm=None,excluded_pairs=()):
        import bm25s
        tokenized=bm25s.tokenize(texts,stopwords="en",show_progress=False)
        # Alias hits are folded to concepts, then capped at channel_k.
        ids,scores=self.engine.retrieve(tokenized,k=min(len(self.codes),k*8),show_progress=False,n_threads=6)
        result=[]
        for row,values in zip(ids,scores):
            folded={}
            for i,s in zip(row,values):
                if exclude_norm is not None and self.aliases.iloc[i].norm_text==exclude_norm:continue
                if excluded_pairs and (self.codes[i],self.aliases.iloc[i].norm_text) in excluded_pairs:continue
                code=self.codes[i]; folded[code]=max(folded.get(code,0),round(float(s),5))
            result.append(sorted(folded.items(),key=lambda x:(-x[1],x[0]))[:k])
        return result


def rrf(channels: dict[str,list], k=60, limit=50):
    scores=defaultdict(float)
    for values in channels.values():
        for rank,(code,_) in enumerate(values,1):scores[code]+=1/(k+rank)
    return sorted(((c,round(s,5)) for c,s in scores.items()),key=lambda x:(-x[1],x[0]))[:limit]


def expand_pool(ranked,system,icd,links,max_k=64,exact=()):
    scores=dict(ranked)
    # Expansion never changes the RRF order of existing candidates.
    codes=[c for c,_ in ranked]
    additions=[]
    if system=="ICD":
        for code in codes[:10]:
            row=icd.get(code)
            if row is None:continue
            parent=row["parent"]
            if parent:additions.append(parent)
            additions.extend(list(row["children"])[:20])
            if parent in icd:additions.extend(list(icd[parent]["children"])[:10])
    else:
        for code in codes[:5]:additions.extend(links.get(code,()))
    # Exact hits are unconditional candidates, without reordering top-1 RRF.
    for code in exact:
        if code not in codes:codes.append(code)
    for code in additions:
        if code not in codes:codes.append(code)
    if len(codes)>max_k:
        must=set(exact)
        survivors=[c for c in codes if c not in must][:max(0,max_k-len(must))]
        codes=[c for c in codes if c in must or c in survivors][:max_k]
    return [(c,scores.get(c,0.0)) for c in codes[:max_k]]
