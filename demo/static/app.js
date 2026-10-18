'use strict';
const $ = id => document.getElementById(id);
const active = new Set(['queued', 'running', 'cancel_requested']);
const stages = ['preparing', 'retrieving', 'reranking', 'deciding', 'validating'];
const labels = { preparing: 'Preparing mention views', retrieving: 'Retrieving candidate codes', reranking: 'Reranking with chart context', deciding: 'Selecting supported codes', validating: 'Validating predictions' };
let job = null, result = null, page = 0, worker = null, uploading = false, submitting = false;
let uploadController = null, selectionVersion = 0, resultLoading = false;
const pageSize = 10;

function announce(text) { $('announcer').textContent = text; }
function showError(message, focus = true) {
  $('error-message').textContent = message; $('error-message').hidden = false;
  $('csv-file').setAttribute('aria-invalid', 'true');
  if (focus) $('error-message').focus();
}
function clearError() { $('error-message').hidden = true; $('error-message').textContent = ''; $('csv-file').removeAttribute('aria-invalid'); }
async function request(url, options = {}) {
  const response = await fetch(url, { ...options, cache: 'no-store' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The request failed. Please try again.');
  return data;
}
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function remember() {
  try { if (job) sessionStorage.setItem('termmap-run', job.id); else sessionStorage.removeItem('termmap-run'); } catch (_) { /* Private browser modes may disable storage. */ }
}
function render() {
  const busy = Boolean(job && active.has(job.status));
  const canRun = Boolean(job && job.status === 'uploaded' && worker && ['ready', 'busy'].includes(worker.status));
  $('run-button').disabled = !canRun || uploading || submitting;
  $('run-label').textContent = uploading ? 'Uploading…' : submitting ? 'Starting…' : busy ? 'Mapping in progress' : 'Run mapping';
  $('cancel-button').hidden = !busy;
  $('cancel-button').disabled = job?.status === 'cancel_requested';
  $('cancel-button').textContent = job?.status === 'cancel_requested' ? 'Cancelling…' : 'Cancel run';
  $('csv-file').disabled = busy || submitting;
  $('sample-button').disabled = busy || uploading || submitting;
  $('remove-file').disabled = busy || uploading || submitting;
  $('drop-zone').classList.toggle('busy', busy || submitting);
  $('file-summary').hidden = !job;
  $('input-preview').hidden = !job || !job.preview;
  $('results-panel').setAttribute('aria-busy', String(busy || uploading));
  if (job) {
    $('file-name').textContent = job.filename; $('file-info').textContent = `${job.rows} clinical ${job.rows === 1 ? 'term' : 'terms'}`;
    if (job.preview) {
      $('preview-rows').replaceChildren(...job.preview.map(row => {
        const div = node('div', undefined, 'preview-row');
        div.append(node('strong', row.clinical_entity), node('p', row.chunk_text || 'No chart context supplied.'));
        return div;
      }));
    }
  }
  const hints = { uploaded: canRun ? 'File checked. Ready to map.' : 'File checked. Waiting for the inference worker.', queued: 'Run queued. This page will update automatically.', running: 'You can leave this tab open while terms are processed.', cancel_requested: 'Finishing the current operation before cancellation.', completed: 'Completed. Choose another file to start a new run.', cancelled: 'Run cancelled. Choose a file to start again.', failed: 'Run stopped. Your uploaded file is retained on the server.' };
  $('run-hint').textContent = uploading ? 'Checking CSV columns and clinical terms…' : job ? hints[job.status] || '' : 'Choose a file to get started.';
  $('run-status').hidden = !job || job.status === 'uploaded';
  $('run-spinner').hidden = !busy;
  $('run-progress').hidden = !busy;
  if (job) {
    $('run-progress').value = Math.round(job.completed / job.rows * 100);
    $('run-status-text').textContent = job.status === 'queued' ? 'Waiting for the inference worker…' : job.status === 'running' ? `${labels[job.stage] || 'Mapping terms'} · ${job.completed} of ${job.rows} terms processed` : job.status === 'completed' ? `Mapping complete · ${job.rows} ${job.rows === 1 ? 'term' : 'terms'} processed` : job.status === 'cancelled' ? 'Run cancelled' : job.status === 'cancel_requested' ? 'Cancellation requested…' : job.status === 'failed' ? 'The mapping run could not finish' : 'File checked and ready.';
    if (job.status === 'failed' && job.error) showError(job.error, false);
  }
  for (const item of $('pipeline-stages').children) {
    const index = stages.indexOf(item.dataset.stage);
    const current = stages.indexOf(job?.stage);
    item.classList.toggle('active', job?.status === 'running' && index === current);
    item.classList.toggle('done', job?.status === 'completed' || (job?.status === 'running' && index < current));
    if (job?.status === 'running' && index === current) item.setAttribute('aria-current', 'step'); else item.removeAttribute('aria-current');
  }
  $('empty-state').hidden = Boolean(result);
  $('results-table-wrap').hidden = !result;
  $('pagination').hidden = !result;
  $('results-footnote').hidden = !result;
  $('download-link').hidden = !result;
  $('result-count').hidden = !result;
}

async function selectFile(file) {
  if (job && active.has(job.status)) return;
  if (!file) return;
  clearError();
  if (!file.name.toLowerCase().endsWith('.csv')) return showError('Choose a .csv file. PDF and spreadsheet files must be exported to CSV first.');
  if (!file.size || file.size > 2 * 1024 * 1024) return showError('Choose a non-empty CSV file up to 2 MB.');
  const version = ++selectionVersion;
  if (uploadController) uploadController.abort();
  uploadController = new AbortController();
  job = null; result = null; remember();
  uploading = true; render();
  try {
    const next = await request('/api/uploads', { method: 'POST', headers: { 'Content-Type': 'text/csv', 'X-Filename': encodeURIComponent(file.name) }, body: file, signal: uploadController.signal });
    if (version !== selectionVersion) return;
    job = next; result = null; page = 0; remember();
    announce(`${job.rows} clinical terms uploaded. Ready to run when inference is available.`);
  } catch (error) {
    if (error.name !== 'AbortError' && version === selectionVersion) showError(error.message);
  } finally {
    if (version === selectionVersion) { uploading = false; render(); }
  }
}

$('csv-file').addEventListener('change', event => selectFile(event.target.files[0]));
$('sample-button').addEventListener('click', async () => {
  clearError();
  try { const response = await fetch('/api/sample.csv'); if (!response.ok) throw new Error('The sample could not be loaded. Try again.'); await selectFile(new File([await response.blob()], 'termmap_sample.csv', { type: 'text/csv' })); } catch (error) { showError(error.message); }
});
$('remove-file').addEventListener('click', () => {
  $('csv-file').value = ''; $('csv-file').click();
});
for (const event of ['dragenter', 'dragover']) $('drop-zone').addEventListener(event, e => { e.preventDefault(); if (!job || !active.has(job.status)) $('drop-zone').classList.add('dragging'); });
for (const event of ['dragleave', 'drop']) $('drop-zone').addEventListener(event, e => { e.preventDefault(); $('drop-zone').classList.remove('dragging'); });
$('drop-zone').addEventListener('drop', e => {
  if (e.dataTransfer.files.length !== 1) return showError('Choose one CSV file at a time.');
  selectFile(e.dataTransfer.files[0]);
});
$('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!job || job.status !== 'uploaded' || submitting) return;
  clearError(); submitting = true; render();
  try { job = await request(`/api/jobs/${job.id}/run`, { method: 'POST' }); remember(); announce('Mapping started.'); } catch (error) { showError(error.message); } finally { submitting = false; render(); }
});
$('cancel-button').addEventListener('click', async () => {
  if (!job || !active.has(job.status)) return;
  try { job = await request(`/api/jobs/${job.id}/cancel`, { method: 'POST' }); render(); announce('Cancellation requested.'); } catch (error) { showError(error.message); }
});

function detailCell(row) {
  const detail = node('details', undefined, 'row-details');
  detail.append(node('summary', `Context & alternatives — ${row.clinical_entity}`));
  const content = node('div', undefined, 'detail-content');
  content.append(node('h3', 'Chart context'), node('p', row.chunk_text || 'No chart context supplied.'));
  const columns = node('div', undefined, 'detail-columns');
  for (const [key, label] of [['ICD', 'ICD-10-CM'], ['SNOMED', 'SNOMED CT']]) {
    const system = row.systems[key], column = node('div');
    column.append(node('h3', `${label} · ${system.candidate_count} candidates`));
    const list = node('ul');
    for (const alternative of system.alternatives) list.append(node('li', `${alternative.code} — ${alternative.description}`));
    column.append(list, node('p', `Model confidence: ${Math.round(system.confidence * 100)}%. This is not a measure of clinical certainty.`));
    const normalized = [...new Set(Object.values(system.views).flat())];
    column.append(node('h3', 'Mention views'), node('p', normalized.join(' · ')), node('h3', 'Supporting retrieval channels'), node('p', system.channels.join(', ') || 'Structural expansion'));
    columns.append(column);
  }
  content.append(columns); detail.append(content); return detail;
}
function renderResults() {
  if (!result) return;
  const total = result.rows.length; page = Math.min(page, Math.max(0, Math.ceil(total / pageSize) - 1));
  const start = page * pageSize, visible = result.rows.slice(start, start + pageSize);
  const body = $('results-body'); body.replaceChildren();
  for (const row of visible) {
    const tr = node('tr'), term = node('td'); term.append(node('span', row.clinical_entity, 'term-name'), node('span', row.chunk_id, 'row-id')); tr.append(term);
    for (const key of ['ICD', 'SNOMED']) {
      const system = row.systems[key], cell = node('td'); cell.append(node('span', system.code, 'code-value'), node('span', system.description, 'code-description')); tr.append(cell);
    }
    const review = Object.values(row.systems).some(system => system.low_confidence);
    const status = node('td'); status.append(node('span', review ? 'Suggested' : '—', `review-badge${review ? '' : ' neutral'}`)); tr.append(status); body.append(tr);
    const details = node('tr', undefined, 'detail-row'), cell = node('td'); cell.colSpan = 4; cell.append(detailCell(row)); details.append(cell); body.append(details);
  }
  $('result-count').textContent = String(total);
  $('page-range').textContent = `${start + 1}–${Math.min(start + pageSize, total)} of ${total} terms`;
  $('previous-page').disabled = page === 0; $('next-page').disabled = start + pageSize >= total;
  $('download-link').href = `/api/jobs/${job.id}/download`; $('download-link').download = 'termmap_predictions.csv';
  $('validation-text').textContent = `Dictionary validation passed · ${Number(result.elapsed).toFixed(1)}s inference · Input order preserved`;
  render();
}
$('previous-page').addEventListener('click', () => { page--; renderResults(); });
$('next-page').addEventListener('click', () => { page++; renderResults(); });

async function poll() {
  try {
    const health = await request('/api/health'); worker = health.worker;
    const names = { ready: 'Inference ready', busy: 'Inference running', loading: 'Loading models…', failed: 'Inference unavailable', offline: 'Inference offline' };
    $('worker-status').textContent = names[worker.status] || 'Connecting…';
    $('worker-status').className = `worker-label ${worker.status}`;
    if (job && active.has(job.status)) {
      const currentId = job.id, next = await request(`/api/jobs/${currentId}`);
      if (job?.id === currentId) job = next;
      if (job?.status === 'completed') announce(`Mapping complete. ${job.rows} terms processed.`);
    }
    if (job?.status === 'completed' && !result && !resultLoading) {
      resultLoading = true; const currentId = job.id;
      try { const data = await request(`/api/jobs/${currentId}/results`); if (job?.id === currentId) { result = data; page = 0; renderResults(); } } finally { resultLoading = false; }
    }
    render();
  } catch (error) {
    $('worker-status').textContent = 'Connection interrupted'; $('worker-status').className = 'worker-label failed';
    worker = null; render();
    if (job && active.has(job.status)) $('run-hint').textContent = 'Connection interrupted. Your run stays on the server; reconnecting automatically…';
  } finally { window.setTimeout(poll, 1500); }
}
async function start() {
  try {
    const id = sessionStorage.getItem('termmap-run');
    if (id) job = await request(`/api/jobs/${id}`);
  } catch (_) { job = null; remember(); }
  render(); poll();
}
start();
