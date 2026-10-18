"""Conservative mention views; chart context gets formatting cleanup only."""
from __future__ import annotations
import html
import math
import re
import unicodedata
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from termmap.preprocessing.noise import PROTECTED, protected


# Match actual markup names, never clinical inequalities such as <0.01 ... >2.
HTML_TAG = re.compile(
    r"</?(?:a|abbr|b|blockquote|body|br|caption|center|code|dd|div|dl|dt|em|font|"
    r"h[1-6]|head|hr|html|i|img|label|li|ol|p|pre|s|small|span|strong|sub|sup|"
    r"table|tbody|td|th|thead|tr|u|ul)\b(?:\s+[^<>]*?)?\s*/?>", re.I)
QUALIFIER_KEYS = {"type", "stage", "grade", "class", "phase"}
# General-language words are not lowercase clinical acronyms merely because a
# broad inventory also lists their capitalized spelling. These are lexical
# safeguards, not row-specific corrections or gold-derived expansions.
ORDINARY_WORDS = set(ENGLISH_STOP_WORDS) | {"got", "gets", "getting", "said", "says", "seen", "went", "done"}
GENERIC_CONTEXT_WORDS = ORDINARY_WORDS | PROTECTED | {
    "patient", "clinical", "disease", "disorder", "finding", "condition", "history", "procedure", "syndrome",
    "test", "testing", "measurement", "assessment", "care", "report", "record", "note", "followup",
}
CUE_PATTERNS = {
    "negation": r"\b(?:no|not|denies|negative for|without|ruled out)\b",
    "history": r"\b(?:h/o|history of)\b",
    "family": r"\b(?:family history|mother|father|sibling)\b",
    "uncertainty": r"\b(?:possible|rule out|suspected)\b",
    "laterality": r"\b(?:left|right|bilateral)\b",
    "acuity": r"\b(?:acute|chronic|subacute)\b",
    "encounter": r"\b(?:initial|subsequent|sequela)\b",
}


def _cues(scope: str, section: str = "") -> dict:
    result = {}
    for name, pattern in CUE_PATTERNS.items():
        match = re.search(pattern, scope, re.I)
        result[name] = match.group(0).lower() if match else ""
    if section in {"pmh", "past medical history"}:
        result["history"] = section
    if section == "family history":
        result["family"] = section
    return result


def _literal_codes(text: str, legal_icd) -> list[str]:
    legal = set(legal_icd)
    # The supplied dictionary is authoritative, including exceptional codes.
    candidates = re.findall(r"(?<!\w)[A-Z][A-Z0-9]{2}(?:\.[A-Z0-9]{1,4})?(?!\w|\.[A-Z0-9])", text.upper())
    return sorted(set(candidates) & legal)


def _qualifier_value(words, index: int) -> bool:
    return index > 0 and words[index-1] in QUALIFIER_KEYS


def clean(text: str) -> str:
    text = html.unescape(unicodedata.normalize("NFKC", text))
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"}))
    text = HTML_TAG.sub(" ", text)
    text = re.sub(r"\(\s*(?:hcc|poa\s*:\s*(?:yes|no))\s*\)", " ", text, flags=re.I)
    # Remove unmatched delimiters, preserve the content and balanced pairs.
    stack, remove = [], set()
    for i, c in enumerate(text):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                stack.pop()
            else:
                remove.add(i)
    remove.update(stack)
    text = "".join(c for i, c in enumerate(text) if i not in remove)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n,;:!?|/\\_\"'")
    # Attached signs can encode measurements or receptor status (ER-/HER2-).
    # Remove decorative dash tokens only when separated from clinical text.
    text = re.sub(r"^-+(?=\s|$)|(?<=\s)-+$", "", text)
    text = re.sub(r"^\.+(?=[^\d.]|$)|\.+$", "", text)
    return text.strip().lower()


def format_context(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = HTML_TAG.sub(" ", text)
    return re.sub(r"\bheader_\d+\s*:\s*", "", text, flags=re.I)


SECTIONS = re.compile(r"(?im)(?:^|\n|\b)(assessment(?:/plan)?|plan|a/p|pmh|past medical history|family history|ros|problem list|hpi|medications|allergies|labs|procedures)\s*:")


def context_features(mention: str, chunk: str, legal_icd=(), window: int = 300) -> dict:
    text = format_context(chunk)
    # Formatting can also affect the mention; no spelling rewrite is applied.
    needle = format_context(mention)
    needle = needle.strip()
    pattern = ((r"(?<!\w)" if needle and needle[0].isalnum() else "")
               + r"\s+".join(re.escape(part) for part in needle.split())
               + (r"(?!\w)" if needle and needle[-1].isalnum() else ""))
    matches = list(re.finditer(pattern, text, re.I)) if needle else []
    if not matches:
        return dict(window=f"[M]{mention}[/M]", no_context=True,
                    cues=_cues(mention), literal_codes=_literal_codes(mention, legal_icd),
                    occurrence_count=0, ambiguous_context=False, selection="mention_only")
    def section_at(position):
        headers = list(SECTIONS.finditer(text, 0, position))
        return headers[-1].group(1).lower() if headers else ""
    # Prefer the assessment/plan location for repeated mentions; never select by
    # negation or gold label. Surface ambiguity explicitly for downstream audits.
    clinical_sections = {"assessment", "assessment/plan", "plan", "a/p", "problem list"}
    match = min(matches, key=lambda m: (section_at(m.start()) not in clinical_sections, m.start()))
    start, end = match.span()
    lo, hi = max(0, start-window), min(len(text), end+window)
    before_boundary = max(text.rfind(". ", lo, start), text.rfind("\n", lo, start))
    if before_boundary >= lo:
        lo = before_boundary+1
    following = re.search(r"[.!?](?:\s|$)|\n", text[end:hi])
    if following:
        hi = end+following.end()
    section = section_at(start)
    result = text[lo:start]+"[M]"+text[start:end]+"[/M]"+text[end:hi]
    if section:
        result = f"[{section}] " + result
    clause = re.split(r"[.;!?,\n]", text[max(0, start-150):start])[-1]
    scope = " ".join(clause.lower().split()[-5:])+" "+mention.lower()
    return dict(window=result, no_context=False, cues=_cues(scope, section),
                literal_codes=_literal_codes(result.replace("[M]", " ").replace("[/M]", " "), legal_icd),
                occurrence_count=len(matches), ambiguous_context=len(matches)>1,
                selection="assessment_plan_then_first")


def normalizer_signature(cfg,out):
    """Fingerprint the inputs to the model-independent final preprocessing cache."""
    from termmap.common.io import sha256
    names=["aliases_final.parquet","abbreviations_final.parquet","abbreviation_alias_support_final.parquet",
           "train.parquet","dev.parquet","sealed_test.parquet"]
    return dict(source_sha256=sha256(cfg["root"]/"termmap"/"preprocessing/normalize.py"),
                noise_sha256=sha256(cfg["root"]/"termmap"/"preprocessing/noise.py"),
                preprocessing=cfg.get("preprocessing",{}),
                inputs={name:sha256(out/name) for name in names})


class Normalizer:
    def __init__(self, aliases, train=None, abbreviations=None, alias_support=None, excluded_names=()):
        excluded_names = set(excluded_names)
        # Preserve the labelled split itself; only omit held-out probe aliases
        # from preprocessing fitting, as the model pair builders already do.
        if train is not None and excluded_names:
            train = train.loc[~train.entity.map(clean).isin(excluded_names)]
        if abbreviations is not None and excluded_names:
            abbreviations = abbreviations.loc[~abbreviations.abbr.isin(excluded_names)
                                               & ~abbreviations.expansion.isin(excluded_names)]
        self.vocab = Counter()
        self.dictionary_vocab = Counter()
        self.dictionary_group_vocab = defaultdict(Counter)
        self.cooccur = defaultdict(set)
        self.spelling_cooccur = defaultdict(set)
        self.dictionary_lexical_tokens = set()
        self.abbreviation_codes = defaultdict(set)
        self.abbr = defaultdict(list)
        self.known_names=set(aliases.norm_text) if "norm_text" in aliases else {clean(t) for t in aliases.text}
        self.dictionary_abbreviations=set()
        self.dictionary_expansions=defaultdict(set)
        ordinary=Counter()
        for text in aliases.text:
            ordinary.update(w for w in re.findall(r"[A-Za-z]+",text) if w.islower())
            m=re.match(r"^([A-Z][A-Z0-9/.-]{1,11})\s+-\s+(.+)$",text)
            if m:
                self.dictionary_abbreviations.add(m.group(1).lower())
                self.dictionary_expansions[m.group(1).lower()].add(clean(m.group(2)))
        self.ordinary_words={w for w,n in ordinary.items() if n>=5}
        self.group_vocab=defaultdict(Counter)
        self.group_only_names=defaultdict(set)
        self.mined_groups=Counter()
        self.tightened=False
        for text in aliases.text:
            tokens = re.findall(r"[a-z]+", clean(text))
            self.vocab.update(tokens)
            self.dictionary_vocab.update(tokens)
            for w in tokens:
                self.cooccur[w].update(tokens)
        # Spelling tie-breaking must not learn associations from group-derived
        # abbreviation aliases. Public terminology is sufficient for this gate.
        public_aliases = aliases.loc[~aliases.source.eq("abbr_variant")] if "source" in aliases else aliases
        for text in public_aliases.text:
            tokens = re.findall(r"[a-z]+", clean(text))
            self.dictionary_lexical_tokens.update(re.findall(r"\b[a-z]+(?:-[a-z]+)+\b",clean(text)))
            for word in tokens:
                self.spelling_cooccur[word].update(tokens)
        if alias_support is not None and len(alias_support):
            self_only={(r.system,r.code,r.norm_text):r.support_groups[0]
                       for r in alias_support.itertuples() if len(r.support_groups)==1}
            totals=Counter(aliases.norm_text);masked=defaultdict(Counter)
            for row in aliases.itertuples():
                group=self_only.get((row.system,row.code,row.norm_text))
                if group is not None:
                    self.group_vocab[group].update(re.findall(r"[a-z]+",clean(row.text)))
                    self.dictionary_group_vocab[group].update(re.findall(r"[a-z]+",clean(row.text)))
                    masked[group][row.norm_text]+=1
            for group,counts in masked.items():
                self.group_only_names[group].update(n for n,count in counts.items() if count==totals[n])
        if train is not None:
            for row in train.itertuples(index=False):
                self.vocab.update(re.findall(r"[a-z]+", (row.entity+" "+row.chunk_text).lower()))
                if hasattr(row,"group_id"):
                    self.group_vocab[row.group_id].update(re.findall(r"[a-z]+",(row.entity+" "+row.chunk_text).lower()))
                    display_words=re.findall(r"[a-z]+",row.display.lower())
                    for token in re.findall(r"\b[a-z]{2,8}\b",row.entity.lower()):
                        if len(display_words)>=2 and "".join(w[0] for w in display_words)==token:
                            self.mined_groups[(token," ".join(display_words),row.group_id)]+=1
        if abbreviations is not None:
            merged=abbreviations.groupby(["abbr","expansion"]).prior_count.sum()
            for (a,e),count in merged.items():self.abbr[a].append((e,int(count),False))
            # Public concept identity can establish that orthographic expansion
            # variants share a sense. Training-derived alias assignments cannot.
            if {"system","code","norm_text"} <= set(public_aliases):
                wanted={e for _,e in merged.index}
                for row in public_aliases.loc[public_aliases.norm_text.isin(wanted)].itertuples():
                    self.abbreviation_codes[row.norm_text].add((row.system,row.code))
        self.prefix_words=defaultdict(list)
        for word in self.vocab:
            if len(word)>=4:self.prefix_words[word[:3]].append(word)
        from symspellpy import SymSpell
        self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for word, frequency in self.vocab.items():
            self.symspell.create_dictionary_entry(word, frequency)

    def _abbreviation_views(self, mention, v1, context, exclude_group):
        words=v1.split();variants=[(v1,0,0,0.0)];decisions=[]
        context_text=clean(re.sub(r"\[/?M\]", " ", context))
        evidence={w for w in re.findall(r"[a-z]+",context_text+" "+v1) if len(w)>=3}-GENERIC_CONTEXT_WORDS
        evidence_text=" ".join(re.findall(r"[a-z0-9]+",context_text+" "+v1))
        for token in dict.fromkeys(words):
            options=[(e,n-self.mined_groups.get((token,e,exclude_group),0)) for e,n,_ in self.abbr.get(token,())]
            options=[(e,n) for e,n in options if n>0]
            if not options:continue
            uppercase=bool(re.search(r"(?<!\w)"+re.escape(token.upper())+r"(?!\w)",mention))
            effective=self.vocab[token]-self.group_vocab.get(exclude_group,{}).get(token,0)
            protected_value=any(i>0 and words[i-1] in QUALIFIER_KEYS for i,w in enumerate(words) if w==token)
            reason=""
            if token in PROTECTED or any(c.isdigit() for c in token) or protected_value or not any(c.isalpha() for c in token):
                reason="protected_token_or_qualifier"
            elif token in ORDINARY_WORDS and not uppercase:reason="ordinary_lowercase_word"
            elif "-" in token and token in self.dictionary_lexical_tokens:reason="known_hyphenated_term"
            elif len(token)>12 or (self.tightened and len(token)>8):reason="not_short_form"
            elif self.tightened and token in self.ordinary_words and not uppercase:reason="ordinary_dictionary_word"
            elif self.tightened and effective>0 and token not in self.dictionary_abbreviations and not uppercase:reason="known_lowercase_word"
            if reason:
                decisions.append(dict(token=token,status="preserved",reason=reason,readings=[]));continue
            # Broad inventories contain contextual paraphrases as well as full
            # forms. A candidate must actually match the short form, or have an
            # explicit public dictionary relationship to it.
            approved=self.dictionary_expansions.get(token,set())
            approved_codes=set().union(*(self.abbreviation_codes.get(e,set()) for e in approved)) if approved else set()
            def plausible(expansion):
                short=re.sub(r"[^a-z]","",token)
                full_words=re.findall(r"[a-z]+",expansion)
                while full_words and full_words[0] in ENGLISH_STOP_WORDS and full_words[0] not in PROTECTED:full_words.pop(0)
                if not short or not full_words or short[0]!=full_words[0][0]:return False
                # Prefixes such as "ECG - no new changes" describe a result;
                # they do not define ECG as "no new changes".
                if expansion in approved or approved_codes & self.abbreviation_codes.get(expansion,set()):return True
                letters=iter("".join(full_words))
                return all(any(c==wanted for c in letters) for wanted in short)
            options=[(e,n) for e,n in options if plausible(e)]
            if not options:
                decisions.append(dict(token=token,status="unresolved",reason="inventory_entries_are_not_supported_full_forms",readings=[]));continue
            keys={tuple(re.findall(r"[a-z0-9]+",e)) for e,_ in options}
            code_sets=[self.abbreviation_codes.get(e,set()) for e,_ in options]
            common=set.intersection(*code_sets) if code_sets and all(code_sets) else set()
            one_sense=len(keys)==1 or bool(common)
            ranked=[]
            for expansion,count in options:
                expansion_words={w for w in re.findall(r"[a-z]+",expansion) if len(w)>=3}-GENERIC_CONTEXT_WORDS
                support=len(expansion_words & (evidence-{token}))
                # Explicit local definitions carry stronger evidence than priors.
                full=re.escape(expansion);short=re.escape(token)
                explicit=bool(re.search(r"(?<!\w)(?:"+full+r"\s*\(\s*"+short+r"\s*\)|"+short+r"\s*\(\s*"+full+r"\s*\))(?!\w)",context_text+" "+v1))
                full_words=" ".join(re.findall(r"[a-z0-9]+",expansion))
                literal=bool(full_words and re.search(r"(?<!\w)"+re.escape(full_words)+r"(?!\w)",evidence_text))
                # A stray shared word (or a single-letter analyte/initial) is
                # not evidence for a clinical sense. Require the full form or
                # all of at least two informative words in the local text.
                supported=support>=2 and support==len(expansion_words)
                if explicit or literal or supported:
                    ranked.append((expansion,count,int(explicit),support))
            if not ranked:
                decisions.append(dict(token=token,status="unresolved",reason="no_context_support_for_reading",readings=[]));continue
            ranked=sorted(ranked,key=lambda x:(-x[2],-x[3],-x[1],x[0]))[:3]
            if any(x[2] for x in ranked):ranked=[x for x in ranked if x[2]]
            new=[]
            for view,definition_score,context_score,prior_score in variants:
                for expansion,count,explicit,support in ranked:
                    pattern=r"(?<!\w)"+re.escape(token)+r"(?!\w)"
                    def replace(match):
                        before=view[:match.start()];after=view[match.end():]
                        # The original full form already supplies the meaning
                        # in a local definition; do not duplicate it.
                        if re.search(r"(?<!\w)"+re.escape(expansion)+r"\s*$",before):return ""
                        if re.search(re.escape(expansion)+r"\s*\(\s*$",before):return ""
                        if re.match(r"\s*\(\s*"+re.escape(expansion)+r"\s*\)",after):return ""
                        parts=expansion.split()
                        for size in range(len(parts),0,-1):
                            suffix=" ".join(parts[-size:])
                            if re.match(r"\s+"+re.escape(suffix)+r"(?!\w)",after):
                                return " ".join(parts[:-size])
                        return expansion
                    expanded=re.sub(pattern,replace,view)
                    expanded=clean(re.sub(r"\(\s*\)"," ",expanded))
                    if re.fullmatch(r"\([^()]+\)",expanded):expanded=expanded[1:-1].strip()
                    new.append((expanded,
                                definition_score+explicit,context_score+support,prior_score+math.log(max(1,count))))
            variants=sorted(new,key=lambda x:(-x[1],-x[2],-x[3],x[0]))[:3]
            decisions.append(dict(token=token,status="supported_alternatives",reason="public_same_sense" if one_sense else "local_definition_or_context",readings=[x[0] for x in ranked]))
        return list(dict.fromkeys(x[0] for x in variants)),decisions

    def abbreviation_diagnostics(self, mention: str, context: str = "", exclude_group=None) -> list[dict]:
        v1=clean(mention)
        if len(v1.split())>40:return [dict(token="",status="preserved",reason="extended_text_span",readings=[])]
        if self.tightened and v1 in self.known_names and v1 not in self.group_only_names.get(exclude_group,()):return []
        return self._abbreviation_views(mention,v1,context,exclude_group)[1]

    def views(self, mention: str, context: str = "", exclude_group=None) -> dict[str, list[str]]:
        v1 = clean(mention)
        words = v1.split()
        effective=lambda w:self.vocab[w]-self.group_vocab.get(exclude_group,{}).get(w,0)
        if len(words)>40 or (self.tightened and v1 in self.known_names and v1 not in self.group_only_names.get(exclude_group,())):
            return dict(V0=[mention],V1=[v1],V2=[v1],V3=[v1],V4=[v1])
        ctx = set(re.findall(r"[a-z]+", context.lower()))
        v2,_ = self._abbreviation_views(mention,v1,context,exclude_group)
        spelling = words.copy()
        from symspellpy import Verbosity
        for i, token in enumerate(words):
            if effective(token)>0 or len(token)<5 or protected(token, self.abbr) or _qualifier_value(words,i):
                continue
            # One-edit, dictionary-supported alternatives only. Never rewrite a
            # known word, a qualifier value, an abbreviation, or chart context.
            suggestions = self.symspell.lookup(token, Verbosity.ALL, max_edit_distance=1)
            if suggestions:
                suggestions=[s for s in suggestions if effective(s.term)>0
                             and self.dictionary_vocab[s.term]-self.dictionary_group_vocab.get(exclude_group,{}).get(s.term,0)>0
                             and not protected(s.term, self.abbr)]
                # Suffixes can encode specimen/analyte identity (e.g. an
                # appended letter), and ordinary words can be one substitution
                # from an unrelated medical word. Do not erase those signals.
                anchors = set(words)-{token}
                suggestions=[s for s in suggestions if len(s.term)>=len(token)
                             and (len(s.term)>len(token) or sorted(s.term)==sorted(token)
                                  or bool(anchors & self.spelling_cooccur[s.term]))]
                if not suggestions:continue
                nearest = min(s.distance for s in suggestions)
                candidates = sorted({s.term for s in suggestions if s.distance == nearest})
                # Resolve multiple candidates only using other mention/context
                # words, and require an unambiguous lexical co-occurrence winner.
                evidence = ctx | (set(words)-{token})
                scores = {term:len(self.spelling_cooccur[term] & evidence) for term in candidates}
                best_score = max(scores.values())
                best = [term for term in candidates if scores[term] == best_score]
                if len(best)==1 and (len(candidates)==1 or best_score>0):
                    spelling[i] = best[0]
        spelling_text = " ".join(spelling)
        if (spelling_text != v1 and (spelling_text not in self.known_names
                or spelling_text in self.group_only_names.get(exclude_group,()))):
            spelling_text = v1
        split = []
        for token_index, token in enumerate(words):
            candidates = []
            inflections=([token[:-1]] if token.endswith("s") else [])+([token[:-2]] if token.endswith("es") else [])+([token[:-3]+"y"] if token.endswith("ies") else [])
            known_inflection=any(self.dictionary_vocab[base]-self.dictionary_group_vocab.get(exclude_group,{}).get(base,0)>0 for base in inflections)
            if effective(token)<=0 and not protected(token, self.abbr) and not _qualifier_value(words,token_index) and not known_inflection:
                for pos in range(3, len(token)-2):
                    a,b = token[:pos], token[pos:]
                    if (effective(a)>0 and effective(b)>0 and a+" "+b in self.known_names
                        and a+" "+b not in self.group_only_names.get(exclude_group,())):
                        candidates.append((min(effective(a), effective(b)), a, b))
            if len(candidates)==1:
                _,a,b = sorted(candidates, key=lambda x:(-x[0],x[1],x[2]))[0]
                split.extend([a,b])
            else:
                split.append(token)
        joined, i = [], 0
        while i<len(split):
            if (i+1<len(split) and (effective(split[i])<=0 or effective(split[i+1])<=0)
                and not protected(split[i], self.abbr) and not protected(split[i+1], self.abbr)
                and not _qualifier_value(split,i) and not _qualifier_value(split,i+1)
                and self.dictionary_vocab[split[i]+split[i+1]]-self.dictionary_group_vocab.get(exclude_group,{}).get(split[i]+split[i+1],0)>0):
                joined.append(split[i]+split[i+1]); i+=2
            else:
                joined.append(split[i]); i+=1
        last=next((i for i in range(len(joined)-1,-1,-1) if joined[i].isalpha()),None)
        if last is not None and len(joined[last])>=3 and effective(joined[last])<=0 and not protected(joined[last], self.abbr) and not _qualifier_value(joined,last):
            prefix = joined[last]
            others = set(joined[:last]+joined[last+1:])
            candidates = [w for w in self.prefix_words.get(prefix[:3],()) if effective(w)>0 and w.startswith(prefix) and w!=prefix
                          and others & self.cooccur[w]
                          and " ".join(joined[:last]+[w]+joined[last+1:]) in self.known_names
                          and " ".join(joined[:last]+[w]+joined[last+1:]) not in self.group_only_names.get(exclude_group,())]
            if len(candidates)==1:
                joined[last] = sorted(candidates, key=lambda w:(-effective(w), w))[0]
        return dict(V0=[mention], V1=[v1], V2=v2, V3=[spelling_text], V4=[" ".join(joined)])
