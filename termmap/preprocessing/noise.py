"""Shared seeded, qualifier-preserving noise for probes, M2 and adaptation."""
from __future__ import annotations
import hashlib
import random
import re

PROTECTED = frozenset("left right bilateral no not without denies negative type stage grade class phase acute chronic subacute initial subsequent sequela mm cm mg mcg kg ml dl l mmol mol iu units percent".split())
NOISE_TYPES = ("merge", "character", "truncate", "case_punctuation", "ehr", "inversion")


def stable_rng(key: str, seed: int = 13) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big"))


def protected(token: str, abbreviations=()) -> bool:
    return (token.lower() in PROTECTED or token.lower() in abbreviations or
            bool(re.search(r"\d", token)) or not token.isalpha())


def corrupt(text: str, kind: str | None = None, seed: int = 13, key: str = "", abbreviations=()) -> tuple[str, str]:
    rng = stable_rng(key or text, seed)
    kind = kind or rng.choice(NOISE_TYPES)
    words = text.split()
    qualifier_values = {i for i in range(1,len(words))
                        if words[i-1].lower() in {"type","stage","grade","class","phase"}}
    editable = [i for i, w in enumerate(words) if i not in qualifier_values
                and not protected(w, abbreviations) and len(w)>=5]
    if kind == "merge":
        pairs = [i for i in range(len(words)-1) if not protected(words[i], abbreviations)
                 and not protected(words[i+1], abbreviations)
                 and i not in qualifier_values and i+1 not in qualifier_values]
        if pairs:
            i = rng.choice(pairs)
            words[i:i+2] = [words[i]+words[i+1]]
    elif kind == "character" and editable:
        i = rng.choice(editable)
        w = words[i]
        j = rng.randrange(1, len(w)-1)
        mode = rng.randrange(3)
        if mode == 0:
            w = w[:j] + w[j+1] + w[j] + w[j+2:]
        elif mode == 1:
            w = w[:j] + w[j+1:]
        else:
            alternatives = [("rn", "m"), ("m", "rn"), ("l", "1"), ("O", "0")]
            a, b = rng.choice(alternatives)
            w = w.replace(a, b, 1) if a in w else w[:j]+w[j+1:]
        words[i] = w
    elif kind == "truncate" and words and len(words)-1 not in qualifier_values and len(words[-1])>=5 and not protected(words[-1], abbreviations):
        words[-1] = words[-1][:-1]
    elif kind == "case_punctuation":
        return text.upper() + ".", kind
    elif kind == "ehr":
        return text + " (hcc)", kind
    elif kind == "inversion" and len(words)>1 and len(words)-1 not in qualifier_values:
        return words[-1]+", "+" ".join(words[:-1]), kind
    return " ".join(words), kind
