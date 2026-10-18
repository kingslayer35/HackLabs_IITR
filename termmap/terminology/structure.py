"""Version 2.1 tree intervals, trusted refinements, families and SQLite cache."""
from __future__ import annotations
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

from termmap.terminology.hierarchy import strip_tag

SPECIFIERS=frozenset("stage grade type class phase mild moderate severe profound acute chronic subacute recurrent persistent intermittent left right bilateral unilateral initial subsequent sequela episode first controlled uncontrolled complicated uncomplicated partial complete".split())
CONTEXT=frozenset("history family no absent without negative not detected suspected risk personal".split())
STOP=frozenset("a an the of in on to for and or with by from at as is are has have structure finding disorder procedure".split())
VALUE=re.compile(r"^(?:\d+[a-z]?|[ivx]+[ab]?|[abc])$")
VALUE_CUES={"stage","grade","type","class","phase","episode"}


def words(text):
    tokens=re.findall(r"[a-z0-9]+",strip_tag(text).lower())
    values={tokens[i+1] for i,t in enumerate(tokens[:-1]) if t in VALUE_CUES}
    return tuple(sorted(set(tokens)-(STOP-values)))


def value_map(text):
    return {k:v for k,v in re.findall(r"\b(stage|grade|type|class|phase)\s+([a-z0-9]+)",strip_tag(text).lower())}


def names(row):
    return sorted({words(row.preferred),words(strip_tag(row.fsn))}-{()})


def core(ws):
    contextual=bool(set(ws)&VALUE_CUES)
    return tuple(w for w in ws if w not in SPECIFIERS|CONTEXT and not (contextual and VALUE.fullmatch(w)))


def seven_definitions(tabular):
    root=ET.parse(tabular).getroot();defs=[];meanings={}
    def visit(node,inherited):
        current=dict(inherited)
        name=node.findtext("name","")
        definition=node.find("sevenChrDef")
        if definition is not None:
            current={e.attrib["char"]:" ".join(e.itertext()).strip() for e in definition.findall("extension")}
            defs.extend(dict(scope_code=name,char=c,meaning=m) for c,m in current.items())
        if name:meanings[name]=current
        for child in node.findall("diag"):visit(child,current)
    for chapter in root.findall("chapter"):
        for section in chapter.findall("section"):
            for diag in section.findall("diag"):visit(diag,{})
    return meanings,pd.DataFrame(defs)


def enrich_icd(frame,tabular,train):
    records=frame.set_index("code").to_dict("index")
    order=sorted(records,key=lambda c:int(records[c]["source_rows"][0]))
    position={c:i for i,c in enumerate(order)}
    spans={}
    def span(code):
        descendants=1;end=position[code]
        for child in records[code]["children"]:
            size,last=span(child);descendants+=size;end=max(end,last)
        if end-position[code]+1!=descendants:raise AssertionError("Noncontiguous ICD subtree")
        spans[code]=(position[code],end,descendants)
        return descendants,end
    for c in order:
        if not records[c]["parent"]:span(c)
    official,defs=seven_definitions(tabular)
    canonical={n.findtext("name",""):n.findtext("desc","") for n in ET.parse(tabular).getroot().iter("diag")}
    official_compact={c.replace(".",""):v for c,v in official.items()}
    freq=Counter(train.loc[train.system.eq("ICD")&train.valid_gold,"code"])
    cumulative=np.r_[0,np.cumsum([freq[c] for c in order])]
    rows=[]
    for c in order:
        r=records[c].copy();parent=records.get(r["parent"])
        description=canonical.get(c,r["preferred"])
        cwords=re.findall(r"[a-z0-9]+",description.lower())
        if parent:
            options=[re.findall(r"[a-z0-9]+",s.lower()) for s in [canonical.get(r["parent"],parent["preferred"]),*parent["descriptions"]]]
            pwords=min(options,key=lambda x:(len(set(cwords)^set(x)),len(x)," ".join(x)))
        else:pwords=[]
        remaining=Counter(pwords);extra=[]
        for w in cwords:
            if remaining[w]:remaining[w]-=1
            elif w not in STOP:extra.append(w)
        compact=c.replace(".","")
        inherited=next((official_compact[compact[:i]] for i in range(len(compact),2,-1) if compact[:i] in official_compact),{})
        meaning=inherited.get(c[-1],"") if len(compact)==7 else ""
        refinement=meaning or " ".join(extra)
        text=refinement.lower()
        kind="type/cause"
        if re.search(r"\b(?:left|right|bilateral|unilateral)\b",text):kind="laterality"
        elif meaning:kind="encounter"
        elif re.search(r"\b(?:stage|grade|class|phase)\b|\b\d+[ab]\b",text):kind="stage"
        elif re.search(r"\b(?:acute|chronic|subacute|recurrent|persistent|intermittent)\b",text):kind="acuity"
        elif re.search(r"\b(?:unspecified|nos)\b",text):kind="unspecified"
        elif re.search(r"\bother\b",text):kind="other"
        elif re.search(r"\b(?:complication|complications|with|without)\b",text):kind="complication"
        elif re.search(r"\b(?:arm|leg|ear|eye|knee|hip|foot|hand|shoulder|ankle|finger|toe|spine|lumbar|cervical|thoracic|forearm|retina|skin|kidney|liver|pulmonary|bone|tendon)\b",text):kind="site"
        a,b,n=spans[c];desc=int(cumulative[b+1]-cumulative[a]-freq[c])
        r.update(code=c,char_len=len(c.replace(".","")),tree_depth=r["depth"],
                 level_name={3:"category",4:"subcategory",5:"sub-subcategory",6:"sub-subcategory",7:"extension"}.get(len(c.replace(".","")),"unknown"),
                 pre_in=a,pre_out=b,n_descendants=n,n_children=len(r["children"]),
                 canonical_description=description,refinement_text=refinement,refinement_kind=kind,seventh_char_meaning=meaning,
                 legacy_gold=len(r["children"])>0 and freq[c]>=2 and freq[c]>desc,
                 header_train_count=freq[c],descendant_train_count=desc)
        rows.append(r)
    result=pd.DataFrame(rows)
    seven=result.loc[result.char_len.eq(7)]
    stats=dict(subtree_contiguous=True,seven_definition_blocks=int(defs.scope_code.nunique()),
               seven_char_codes=len(seven),seven_meaning_coverage=float(seven.seventh_char_meaning.ne("").mean()),
               header_exceptions=result.loc[result.legacy_gold,"code"].tolist(),levels=result.level_name.value_counts().to_dict())
    return result,defs,stats


def snomed_links(concepts):
    by_name=defaultdict(set);by_core=defaultdict(set);tag={};contextual={};value_maps=defaultdict(list)
    for r in concepts.itertuples():
        tag[r.code]=r.tag
        contextual[r.code]=bool(r.history or r.family_history or r.absent)
        if r.tag=="unknown":continue
        for ws in names(r):
            by_name[(r.tag,ws)].add(r.code)
            for text in [r.preferred,strip_tag(r.fsn)]:
                if words(text)==ws:value_maps[(r.code,ws)].append(value_map(text))
            key=core(ws)
            if key:by_core[(r.tag,key)].add(r.code)
    edges=set()
    for (t,key),codes in by_core.items():
        codes=sorted(codes)
        for src in codes:
            for dst in codes:
                if src!=dst:edges.add((src,dst,"qualifier_sibling"))
        if t=="situation":
            for dst in by_core.get(("finding",key),set())|by_core.get(("disorder",key),set()):
                for src in codes:
                    if contextual[src]:
                        edges.add((src,dst,"qualifier_sibling"));edges.add((dst,src,"qualifier_sibling"))
    for (t,ws),children in by_name.items():
        candidate_forms=[(ws,())]
        if set(ws)&VALUE_CUES:
            for i,w in enumerate(ws):
                if re.fullmatch(r"\d+[a-z]|[ivx]+[ab]",w):
                    candidate_forms.append((tuple(sorted((*ws[:i],w[:-1],*ws[i+1:]))),(w,)))
        for form,refined in candidate_forms:
            for n in range(0 if refined else 1,3-len(refined)):
                for deleted in combinations(range(len(form)),n):
                    parent_words=tuple(w for i,w in enumerate(form) if i not in deleted)
                    if not parent_words:continue
                    parents=by_name.get((t,parent_words),())
                    if not parents:continue
                    extra=tuple(form[i] for i in deleted)+refined
                    value_allowed=bool(set(ws)&VALUE_CUES)
                    trusted=all(w in SPECIFIERS or (value_allowed and VALUE.fullmatch(w)) for w in extra)
                    kind="specifier_child" if trusted else "lookalike"
                    for src in parents:
                        for dst in children:
                            compatible=any(all(k not in cm or cm[k]==v or (cm[k].startswith(v) and cm[k][len(v):] in {"a","b","c"}) for k,v in pm.items()) for pm in value_maps[(src,parent_words)] for cm in value_maps[(dst,ws)])
                            actual=kind if compatible else "lookalike"
                            if src!=dst:edges.add((src,dst,actual))
    # If another name establishes a trusted relation, retain that stronger kind.
    trusted={(a,b) for a,b,k in edges if k=="specifier_child"}
    edges=[(a,b,k) for a,b,k in edges if k!="lookalike" or (a,b) not in trusted]
    return pd.DataFrame(sorted(edges),columns=["src","dst","kind"])


def family_index(concepts,links):
    parent={c:c for c in concepts.code}
    def find(c):
        while parent[c]!=c:parent[c]=parent[parent[c]];c=parent[c]
        return c
    for r in links.loc[links.kind.eq("specifier_child")].itertuples():
        a,b=find(r.src),find(r.dst);parent[max(a,b)]=min(a,b)
    concepts=concepts.copy();concepts["family_id"]=[find(c) for c in concepts.code]
    concepts=concepts.sort_values(["top_level","tag","family_id","code"]).reset_index(drop=True)
    concepts["family_position"]=np.arange(len(concepts))
    families=concepts.groupby("family_id",sort=False).family_position.agg(["min","max","count"]).reset_index().rename(columns={"min":"start","max":"end","count":"size"})
    return concepts,families


def save_sqlite(path,icd,snomed,links,families,seventh,aliases):
    def serializable(frame):
        frame=frame.copy()
        for col in frame:
            if any(isinstance(v,(list,dict,np.ndarray)) for v in frame[col].head(10)):
                frame[col]=frame[col].map(lambda v:json.dumps(v.tolist() if isinstance(v,np.ndarray) else v))
        return frame
    with sqlite3.connect(path) as connection:
        def primary_table(frame,table,key):
            frame=serializable(frame)
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            definitions=[]
            for column in frame:
                dtype="INTEGER" if frame[column].dtype.kind in "biu" else "REAL" if frame[column].dtype.kind=="f" else "TEXT"
                definitions.append(f'"{column}" {dtype}'+(" PRIMARY KEY" if column==key else ""))
            connection.execute(f'CREATE TABLE "{table}" ('+", ".join(definitions)+")")
            frame.to_sql(table,connection,if_exists="append",index=False)
        primary_table(icd,"icd_node","code")
        aliases.loc[aliases.system.eq("ICD"),["code","text","source"]].to_sql("icd_name",connection,if_exists="replace",index=False)
        seventh.to_sql("icd_seventh_def",connection,if_exists="replace",index=False)
        primary_table(snomed,"snomed_concept","code")
        links.to_sql("snomed_link",connection,if_exists="replace",index=False)
        primary_table(families,"snomed_family","family_id")
        for table,columns in {"icd_node":["code","parent","pre_in","category","block"],"icd_name":["code"],"snomed_concept":["code","tag","top_level","family_id"],"snomed_link":["src","dst"],"snomed_family":["family_id"]}.items():
            for col in columns:connection.execute(f'CREATE INDEX IF NOT EXISTS ix_{table}_{col} ON {table}("{col}")')


class TerminologyStore:
    """Load one SQLite snapshot once; every query afterwards is in memory."""
    def __init__(self,path):
        with sqlite3.connect(path) as connection:
            self.icd=pd.read_sql_query("SELECT * FROM icd_node ORDER BY pre_in",connection)
            self.snomed=pd.read_sql_query("SELECT * FROM snomed_concept ORDER BY family_position",connection)
            self.links=pd.read_sql_query("SELECT * FROM snomed_link",connection)
            self.families=pd.read_sql_query("SELECT * FROM snomed_family",connection)
        self.nodes=self.icd.set_index("code").to_dict("index")

    def subtree(self,code):
        n=self.nodes[code];return self.icd.iloc[n["pre_in"]:n["pre_out"]+1]

    def contains(self,parent,child):
        a,b=self.nodes[parent],self.nodes[child]
        return a["pre_in"]<=b["pre_in"]<=a["pre_out"]
