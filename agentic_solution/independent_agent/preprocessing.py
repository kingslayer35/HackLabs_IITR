"""Conservative, label-free mention views and raw-context diagnostics.

Dictionary-derived abbreviations remain ambiguous; expansion never replaces raw.
This first repair pass permits unique single-edit corrections and two-word splits,
which is deliberately narrower than the team's proposed edit-distance-two pass.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import html
import re
import unicodedata


PROTECTED = set('no not without denies denied negative absent absence left right bilateral acute chronic type stage grade history family initial subsequent sequela mild moderate severe'.split())
COMMON = set('a an as at be by for from in is it of on or the to with and us i ii iii iv v vi vii viii ix x'.split())
TAGS = re.compile(r'\s*\((?:disorder|finding|procedure|situation|observable entity|body structure|morphologic abnormality|qualifier value|substance|organism|physical object|regime/therapy|event|occupation|environment|assessment scale|navigational concept|record artifact|person|clinical drug|medicinal product|product)\)\s*$',re.I)
TOKEN = re.compile(r'\b[A-Za-z][A-Za-z0-9]*\b')


def clean_mention(text):
    text=unicodedata.normalize('NFKC',html.unescape(str(text)))
    text=text.translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','–':'-','—':'-'}))
    text=re.sub(r'<br\s*/?>',' ',text,flags=re.I)
    text=re.sub(r'<[^>]+>',' ',text)
    text=re.sub(r'\bheader_\d+\s*:\s*',' ',text,flags=re.I)
    text=re.sub(r'\(\s*(?:hcc(?:\s+\d+)?|poa\s*:\s*(?:yes|no|y|n))\s*\)',' ',text,flags=re.I)
    return ' '.join(text.casefold().split())


def single_edit(a,b):
    if a==b:return False
    if abs(len(a)-len(b))>1:return False
    if len(a)==len(b):
        mismatches=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
        return len(mismatches)==1 or (len(mismatches)==2 and mismatches[1]==mismatches[0]+1 and a[mismatches[0]]==b[mismatches[1]] and a[mismatches[1]]==b[mismatches[0]])
    if len(a)>len(b):a,b=b,a
    i=j=errors=0
    while i<len(a) and j<len(b):
        if a[i]==b[j]:i+=1;j+=1
        else:
            errors+=1;j+=1
            if errors>1:return False
    return True


class Preprocessor:
    def __init__(self, aliases):
        self.vocab=Counter(); self.ordinary_words=Counter(); self.abbreviations=defaultdict(Counter)
        for alias in aliases:
            text=TAGS.sub('',str(alias)).strip()
            self.vocab.update(re.findall(r'\b[a-z]{2,}\b',text.casefold()))
            self.ordinary_words.update(t.casefold() for t in re.findall(r'\b[A-Za-z]{3,}\b',text) if not t.isupper())
            candidates=[]
            m=re.match(r'^([A-Z][A-Z0-9/]{1,11})\s+[-–—]\s+(.+)$',text)
            if m:candidates.append((m[1],m[2]))
            m=re.match(r'^(.+?)\s+\(([A-Z][A-Z0-9]{1,11})\)$',text)
            if m:candidates.append((m[2],m[1]))
            for abbr,expansion in candidates:
                key=abbr.casefold(); value=clean_mention(expansion)
                if key in COMMON or key in PROTECTED or len(value)<len(key)+2:continue
                if len(value)>150 or not re.search('[a-z]',value):continue
                words=re.findall(r'[a-z]+',value)
                letters=''.join(re.findall('[a-z]',key))
                initials=''.join(w[0] for w in words if w not in {'of','and','by','for','with','the'})
                if len(words)==1:
                    cursor=iter(words[0])
                    aligned=all(any(c==letter for c in cursor) for letter in letters)
                else:aligned=initials==letters
                if not aligned:continue
                self.abbreviations[key][value]+=1
        # Explicit known coverage gaps named in the team plan. These are query
        # alternatives, not new dictionary aliases or code assignments.
        for abbr,expanded in [('pna','pneumonia'),('doe','dyspnea on exertion')]:
            self.abbreviations[abbr][expanded]+=1
        # Index vocabulary deletion signatures (SymSpell-style, one edit).
        self.deletions=defaultdict(set)
        for word,count in self.vocab.items():
            if count<5 or len(word)<5:continue
            for i in range(len(word)):
                self.deletions[word[:i]+word[i+1:]].add(word)

    @lru_cache(maxsize=10000)
    def repair_word(self,word):
        if not word.isalpha() or len(word)<5 or word in self.vocab or word in PROTECTED or word in self.abbreviations:
            return word,None
        splits=[(word[:i],word[i:]) for i in range(3,len(word)-2)
                if self.vocab[word[:i]]>=5 and self.vocab[word[i:]]>=5]
        if len(splits)==1:return ' '.join(splits[0]),'split'
        candidates=set(self.deletions.get(word,()))
        for i in range(len(word)):
            deletion=word[:i]+word[i+1:]
            candidates.update(self.deletions.get(deletion,()))
            if self.vocab[deletion]>=5:candidates.add(deletion)
        for i in range(len(word)-1):
            swapped=word[:i]+word[i+1]+word[i]+word[i+2:]
            if self.vocab[swapped]>=5:candidates.add(swapped)
        def safe_edit(candidate):
            if not single_edit(word,candidate):return False
            # Allow missing-letter insertion, adjacent transposition, or removing
            # an accidental repeated letter. Do not substitute one letter or
            # delete an arbitrary prefix/suffix from a potentially valid word.
            if len(candidate)>len(word):return True
            if len(candidate)==len(word):
                return sum(a!=b for a,b in zip(word,candidate))==2
            return any(word[i]==word[i+1] and word[:i]+word[i+1:]==candidate for i in range(len(word)-1))
        candidates={c for c in candidates if c not in PROTECTED and c not in self.abbreviations and safe_edit(c)}
        if len(candidates)==1:return next(iter(candidates)),'spelling'
        return word,None

    def views(self,mention):
        clean=clean_mention(mention)
        result=[dict(kind='raw',text=mention)]
        if clean!=mention:result.append(dict(kind='clean',text=clean))
        # A bounded beam expands all eligible tokens without choosing an
        # abbreviation sense from chart context. No label-derived priors.
        beam=[clean]; expansions=[]
        for token in dict.fromkeys(TOKEN.findall(clean)):
            options=self.abbreviations.get(token.casefold())
            if not options or token in PROTECTED or token in COMMON:continue
            if len(token)>=3 and self.ordinary_words[token]>=5:continue
            choices=sorted(options,key=lambda x:(-options[x],x))[:3]
            proposals=[]
            for base in beam:
                for choice in choices:
                    updated=re.sub(r'\b'+re.escape(token)+r'\b',lambda _:choice,base,flags=re.I)
                    if len(updated)<=500:proposals.append(updated)
            if proposals:
                beam=list(dict.fromkeys(proposals))[:3]
                expansions.append(dict(token=token,choices=choices))
        if expansions:
            result.extend(dict(kind='abbreviation',text=x) for x in beam if x!=clean)
        repairs=[]
        def fix(m):
            old=m[0]
            if clean[m.end():m.end()+2]=="'s":return old
            new,kind=self.repair_word(old)
            if kind:repairs.append(dict(source=old,replacement=new,kind=kind))
            return new
        repaired=TOKEN.sub(fix,clean)
        if repaired!=clean:result.append(dict(kind='repair',text=repaired))
        seen=set();dedup=[]
        for view in result:
            if view['text'] not in seen:dedup.append(view);seen.add(view['text'])
        return dict(views=dedup,expansions=expansions,repairs=repairs)


def context_diagnostics(mention,chunk):
    # Preserve chart wording and avoid relocating the mention after cleaning.
    if not chunk:return dict(window=mention,no_context=True,mention_found=False,cues={})
    pos=chunk.casefold().find(mention.casefold())
    found=pos>=0
    if not found:return dict(window=chunk[:600],no_context=False,mention_found=False,cues={})
    start=max(0,pos-300);end=min(len(chunk),pos+len(mention)+300)
    window=chunk[start:pos]+'[M]'+chunk[pos:pos+len(mention)]+'[/M]'+chunk[pos+len(mention):end]
    cues={name:bool(re.search(pattern,window,re.I)) for name,pattern in dict(
        negation=r'\b(?:no|denies|without|negative for)\b',history=r'\b(?:history of|h/o|previous)\b',
        family=r'\bfamily history\b',uncertainty=r'\b(?:possible|suspected|rule out)\b',
        laterality=r'\b(?:left|right|bilateral)\b',acuity=r'\b(?:acute|chronic)\b',
        encounter=r'\b(?:initial|subsequent|sequela)\b').items()}
    return dict(window=window,no_context=False,mention_found=True,cues=cues,
                note='Diagnostic cue presence only, not assertion scope or coding policy; not used in retrieval.')
