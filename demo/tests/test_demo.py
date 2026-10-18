import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from demo.server import make_handler
from demo.store import JobStore, MAX_BYTES, parse_csv, write_json


CSV = b'chunk_id,clinical_entity,chunk_text,code\n0001,"right knee pain","Chart, with context",GOLD_MUST_NOT_REACH_MODEL\n'


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = JobStore(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_string_ids_context_and_gold_exclusion(self):
        rows = parse_csv(CSV)
        self.assertEqual(rows[0]['chunk_id'], '0001')
        self.assertEqual(rows[0]['chunk_text'], 'Chart, with context')
        self.assertNotIn('code', rows[0])
        self.assertEqual(parse_csv(b'entity\nMS\n')[0]['clinical_entity'], 'MS')

    def test_csv_validation(self):
        for content in [b'', b'wrong\nvalue\n', b'entity\n\n', b'entity\n""\n', b'entity,entity\na,b\n', b'entity,chunk_text\na\n', b'entity\na,b\n', b'entity\n\xff', b'entity\n\x00', b'entity\n'+b'a\n'*101, b'x'*(MAX_BYTES+1)]:
            with self.subTest(content=content[:25]), self.assertRaises(ValueError):
                parse_csv(content)

    def test_no_traversal_and_upload_filename_sanitized(self):
        job = self.store.create(CSV, '../../private/terms.csv')
        self.assertEqual(job['filename'], 'terms.csv')
        for bad in ['../x', 'a'*31, 'not-a-uuid', '/etc/passwd']:
            with self.assertRaises(KeyError):
                self.store.get(bad)

    def test_duplicate_submission_single_claim(self):
        job = self.store.create(CSV, 'terms.csv')
        self.store.enqueue(job['id'])
        self.store.enqueue(job['id'])
        self.assertEqual(self.store.claim()['id'], job['id'])
        self.assertIsNone(self.store.claim())

    def test_cancel_before_and_during_inference(self):
        job = self.store.create(CSV, 'terms.csv'); self.store.enqueue(job['id'])
        self.store.cancel(job['id']); self.assertIsNone(self.store.claim())
        job = self.store.create(CSV, 'terms.csv'); self.store.enqueue(job['id']); self.store.claim()
        self.store.cancel(job['id'])
        with self.assertRaises(InterruptedError): self.store.progress(job['id'], 'reranking', 0)
        self.store.finish(job['id'], 'completed')
        self.assertEqual(self.store.get(job['id'])['status'], 'cancelled')

    def test_queue_capacity_and_recovery(self):
        for _ in range(4): self.store.enqueue(self.store.create(CSV, 'terms.csv')['id'])
        with self.assertRaises(ValueError): self.store.enqueue(self.store.create(CSV, 'terms.csv')['id'])
        running = self.store.claim(); self.store.recover()
        self.assertEqual(self.store.get(running['id'])['status'], 'failed')

    def test_worker_heartbeat(self):
        self.assertEqual(self.store.worker_state()['status'], 'offline')
        write_json(Path(self.temp.name)/'worker.json', dict(status='ready', heartbeat=time.time()))
        self.assertEqual(self.store.worker_state()['status'], 'ready')
        write_json(Path(self.temp.name)/'worker.json', dict(status='ready', heartbeat=0))
        self.assertEqual(self.store.worker_state()['status'], 'offline')


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.store = JobStore(self.temp.name)
        self.server = ThreadingHTTPServer(('127.0.0.1',0), make_handler(self.store))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.base = 'http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.temp.cleanup()

    def request(self, path, data=None, headers=None):
        req = Request(self.base+path, data=data, headers=headers or {})
        try:
            with urlopen(req,timeout=5) as response: return response.status, response.read(), response.headers
        except HTTPError as error: return error.code, error.read(), error.headers

    def upload(self):
        code, raw, _ = self.request('/api/uploads', CSV, {'Content-Type':'text/csv','X-Filename':'terms.csv'})
        self.assertEqual(code,201); return json.loads(raw)

    def ready(self):
        write_json(Path(self.temp.name)/'worker.json',dict(status='ready',heartbeat=time.time()))

    def test_upload_queue_real_result_download_contract(self):
        job=self.upload(); job_id=job['id']; self.assertEqual(job['preview'][0]['chunk_id'],'0001')
        self.assertEqual(self.request(f'/api/jobs/{job_id}/run',b'')[0],409)
        self.ready(); self.assertEqual(self.request(f'/api/jobs/{job_id}/run',b'')[0],202)
        self.assertEqual(self.request(f'/api/jobs/{job_id}/run',b'')[0],202)
        self.assertEqual(self.request(f'/api/jobs/{job_id}/download')[0],409)
        self.store.claim()
        folder=self.store.folder(job_id)
        # Explicit test fixture: HTTP transport checks, not live model predictions.
        (folder/'predictions.csv').write_text('chunk_id,clinical_entity,icd10cm_code,snomed_code\n0001,right knee pain,M25.561,TEST_ONLY\n')
        write_json(folder/'results.json',dict(rows=[],validator='passed'))
        self.store.finish(job_id,'completed')
        code,raw,headers=self.request(f'/api/jobs/{job_id}/download')
        self.assertEqual(code,200); self.assertIn(b'0001',raw); self.assertIn('attachment',headers['Content-Disposition'])
        self.assertEqual(self.request(f'/api/jobs/{job_id}/results')[0],200)

    def test_validation_security_and_static_routes(self):
        self.assertEqual(self.request('/api/uploads',b'entity\nMS\n',{'Content-Type':'text/plain'})[0],415)
        self.assertEqual(self.request('/api/uploads',b'wrong\na\n',{'Content-Type':'text/csv'})[0],400)
        self.assertEqual(self.request('/api/uploads',CSV,{'Content-Type':'text/csv','Origin':'https://other.invalid'})[0],403)
        self.assertEqual(self.request('/api/jobs/../../secret')[0],404)
        self.assertEqual(self.request('/not-a-page')[0],404)
        code,body,headers=self.request('/')
        self.assertEqual(code,200); self.assertIn(b'novalidate',body)
        self.assertEqual(headers['Cache-Control'],'no-store'); self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy'])

    def test_cancelled_outputs_are_unavailable(self):
        self.ready(); job=self.upload(); job_id=job['id']
        self.request(f'/api/jobs/{job_id}/run',b'')
        self.assertEqual(self.request(f'/api/jobs/{job_id}/cancel',b'')[0],200)
        self.assertEqual(self.request(f'/api/jobs/{job_id}/download')[0],409)


if __name__ == '__main__': unittest.main()
