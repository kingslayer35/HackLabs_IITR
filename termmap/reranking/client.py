"""Cache exact packed token inputs while retaining the frozen scorer contract."""
import hashlib
import json
from pathlib import Path
import sqlite3

from termmap.reranking.cards import CachedCards
from termmap.common.io import sha256
from termmap.reranking.vllm import VllmRanker


class InputCachedRanker(VllmRanker):
    def __init__(self, cfg, adapter=False, cards=None):
        super().__init__(cfg, adapter=adapter, cards=cards if cards is not None else CachedCards(cfg))
        self.input_connection = None
        try:
            # Identity already pins model/tokenizer/template, dtype and GPU.
            # The input-cache implementation is also pinned independently.
            name = self.identity + "_inputs_" + sha256(Path(__file__))[:16] + ".sqlite"
            self.input_connection = sqlite3.connect(self.runs / "reranker_cache" / name, timeout=30)
            self.input_connection.execute("CREATE TABLE IF NOT EXISTS inputs (key TEXT PRIMARY KEY,tokens TEXT NOT NULL)")
        except BaseException:
            super().close()
            raise

    def token_ids(self, payloads):
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in payloads]
        known = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0,len(unique),500):
            selected = unique[start:start+500]
            entries = self.input_connection.execute(
                "SELECT key,tokens FROM inputs WHERE key IN (" + ",".join("?" for _ in selected) + ")", selected).fetchall()
            known.update((key,json.loads(tokens)) for key,tokens in entries)
        missing = {}
        for key,payload in zip(keys,payloads):
            if key not in known:missing.setdefault(key,payload)
        if missing:
            ordered = list(missing)
            # The original packing code remains the only token encoder.
            tokens = super().token_ids([missing[key] for key in ordered])
            known.update(zip(ordered,tokens))
            self.input_connection.executemany("INSERT OR IGNORE INTO inputs VALUES (?,?)",
                [(key,json.dumps(value,separators=(",",":"))) for key,value in zip(ordered,tokens)])
            self.input_connection.commit()
        return [known[key] for key in keys]

    def close(self):
        try:
            super().close()
        finally:
            if self.input_connection is not None:
                self.input_connection.close()
                self.input_connection = None


def make_ranker(cfg,adapter=False,cards=None):
    cls = InputCachedRanker if cfg.get("vllm",{}).get("input_cache_enabled",False) else VllmRanker
    return cls(cfg,adapter=adapter,cards=cards)
