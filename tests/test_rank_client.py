"""Input caching preserves the original codec and ordering, including masks."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from termmap.reranking.client import InputCachedRanker
from termmap.reranking.vllm import VllmRanker


def test_input_cache_delegates_only_misses_and_reopens_without_reencoding(tmp_path,monkeypatch):
    encoded=[]
    def canonical(self,payloads):
        encoded.extend(payloads)
        return [[13]+[ord(char) for char in value]+[17] for value in payloads]
    monkeypatch.setattr(VllmRanker,"token_ids",canonical)
    ranker=object.__new__(InputCachedRanker)
    file=tmp_path/"input.sqlite"
    ranker.input_connection=sqlite3.connect(file)
    ranker.input_connection.execute("CREATE TABLE inputs (key TEXT PRIMARY KEY,tokens TEXT NOT NULL)")
    texts=["query α | held alias removed","query β | training group excluded","query α | held alias removed"]
    expected=canonical(ranker,texts);encoded.clear()
    assert ranker.token_ids(texts)==expected
    assert encoded==texts[:2]
    ranker.input_connection.close()
    ranker.input_connection=sqlite3.connect(file)
    encoded.clear()
    assert ranker.token_ids(list(reversed(texts)))==list(reversed(expected))
    assert not encoded
    # Distinct masked text cannot reuse another mask's cached input.
    changed=["query α | explicit alias withheld",texts[1]]
    observed=ranker.token_ids(changed)
    assert encoded==changed[:1]
    assert observed==canonical(ranker,changed)
    assert ranker.token_ids([])==[]
    ranker.input_connection.close()


def test_concurrent_input_cache_writers_preserve_exact_order_and_unique_inputs(tmp_path,monkeypatch):
    def canonical(self,payloads):return [[13]+[ord(char) for char in value]+[17] for value in payloads]
    monkeypatch.setattr(VllmRanker,"token_ids",canonical)
    file=tmp_path/"shared.sqlite";barrier=Barrier(2)
    connection=sqlite3.connect(file)
    connection.execute("CREATE TABLE inputs (key TEXT PRIMARY KEY,tokens TEXT NOT NULL)");connection.close()
    texts=["public mention A","public mention B","public mention A","explicit mask C"]
    def worker(reverse):
        ranker=object.__new__(InputCachedRanker);ranker.input_connection=sqlite3.connect(file,timeout=30)
        values=texts[::-1] if reverse else texts
        barrier.wait()
        try:
            result=ranker.token_ids(values)
            assert result==canonical(ranker,values)
            return result[::-1] if reverse else result
        finally:ranker.input_connection.close()
    with ThreadPoolExecutor(2) as executor:
        left=executor.submit(worker,False);right=executor.submit(worker,True)
        assert left.result()==right.result()
    connection=sqlite3.connect(file)
    assert connection.execute("SELECT COUNT(*) FROM inputs").fetchone()[0]==3
    connection.close()
