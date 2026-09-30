/* ==========================================================================
   Denoise & Separate -- frontend controller (vanilla JS, no build step)

   Flow:  pick file -> tick paths -> POST /api/jobs -> poll -> results appear
   one by one as the backend writes them, then the full report + comparison.
   ========================================================================== */

'use strict';

const API = '';
const POLL_MS = 700;
const SPEAKER_COLORS = ['#5eead4', '#7c9cff', '#f0abfc', '#fbbf24', '#4ade80', '#fb7185', '#38bdf8', '#c084fc'];

const state = {
  file: null,
  methods: { denoisers: [], separators: [] },
  paths: [],
  selected: new Set(['path1']),
  custom: [],            // [{id, denoiser, separator}] -- the builder rows
  rowSeq: 0,             // counter behind the custom_<n> row ids
  job: null,
  seq: 0,
  cards: new Map(),      // path_id -> DOM node
  timer: null,
  startedAt: 0,
  polling: null,
  health: null,
};

/* ------------------------------------------------------------------ utils */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, cls, html) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function esc(str) {
  return String(str == null ? '' : str).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function fmtTime(seconds) {
  const s = Number(seconds) || 0;
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(2)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

function fmtClock(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

function fmtSize(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let n = Number(bytes) || 0, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${i === 0 ? n : n.toFixed(1)} ${units[i]}`;
}

function num(value, digits = 1, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function toast(message, kind = 'info', ttl = 6000) {
  const node = el('div', `toast ${kind}`, esc(message));
  $('#toasts').appendChild(node);
  setTimeout(() => { node.style.opacity = '0'; setTimeout(() => node.remove(), 300); }, ttl);
}

/* --------------------------------------------------------------- waveform */
function drawWave(canvas, peaks, color) {
  if (!canvas || !peaks || !peaks.length) return;
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.clientHeight || 76;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const mid = height / 2;
  const step = width / peaks.length;
  const grad = ctx.createLinearGradient(0, 0, width, 0);
  grad.addColorStop(0, color || '#5eead4');
  grad.addColorStop(1, '#7c9cff');
  ctx.fillStyle = grad;

  for (let i = 0; i < peaks.length; i += 1) {
    const h = Math.max(1.2, peaks[i] * (height * 0.92));
    const x = i * step;
    ctx.fillRect(x, mid - h / 2, Math.max(1, step * 0.7), h);
  }
  ctx.strokeStyle = 'rgba(255,255,255,.06)';
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(width, mid); ctx.stroke();
}

async function localWaveform(file, points = 500) {
  try {
    const buf = await file.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const decoded = await ctx.decodeAudioData(buf.slice(0));
    const data = decoded.getChannelData(0);
    const block = Math.max(1, Math.floor(data.length / points));
    const peaks = [];
    for (let i = 0; i < points; i += 1) {
      let peak = 0;
      const start = i * block;
      for (let j = start; j < start + block && j < data.length; j += 1) {
        const v = Math.abs(data[j]);
        if (v > peak) peak = v;
      }
      peaks.push(peak);
    }
    const top = Math.max(...peaks) || 1;
    ctx.close();
    return {
      peaks: peaks.map((p) => p / top),
      duration: decoded.duration,
      sampleRate: decoded.sampleRate,
      channels: decoded.numberOfChannels,
    };
  } catch (err) {
    return null;
  }
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  initTheme();
  wireUpload();
  wireControls();
  await Promise.all([loadHealth(), loadMethodsAndPaths()]);
  restoreSelection();
  updateRunState();
}

function initTheme() {
  const saved = localStorage.getItem('ds-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('ds-theme', next);
    redrawAllWaves();
  });
}

async function loadHealth() {
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    const dot = $('#health .dot');
    dot.className = 'dot ok';
    $('#health-text').textContent =
      `${data.denoisers_available}/${data.denoisers_total} denoisers · ` +
      `${data.separators_available}/${data.separators_total} separators` +
      `${data.torch ? '' : ' · no torch'}`;
    $('#health').title =
      `python ${data.python} · ${data.platform}\nmax upload ${data.max_upload_mb} MB` +
      `\nffmpeg: ${data.ffmpeg ? data.ffmpeg_path : 'not found'}` +
      `\nHF token: ${data.hf_token_set ? 'set' : 'not set'}`;
    state.health = data;
    renderEnvBanner(data);
  } catch (err) {
    $('#health .dot').className = 'dot bad';
    $('#health-text').textContent = 'backend unreachable';
  }
}

/**
 * One clear explanation of what is missing, instead of the same package error
 * repeated on every blocked card.
 */
function renderEnvBanner(health) {
  const box = $('#env-banner');
  if (!box) return;
  const notes = [];

  if (!health.torch) {
    const py = String(health.python || '');
    const newPython = /^3\.(1[3-9]|[2-9]\d)/.test(py);
    notes.push({
      title: 'Neural methods are switched off (PyTorch is not installed)',
      body: newPython
        ? `The DSP methods below work fine. For DeepFilterNet, SepFormer, pyannote and the rest you need PyTorch — but there are no PyTorch wheels for Python ${py} yet, so <code>pip install torch</code> will fail on this interpreter. Create a Python 3.11 environment for those backends.`
        : 'The DSP methods below work fine. Install PyTorch to unlock DeepFilterNet, SepFormer, pyannote and the other neural backends.',
      cmd: newPython
        ? 'py -3.11 -m venv .venv311 &amp;&amp; .venv311\\Scripts\\pip install -r requirements.txt torch torchaudio deepfilternet speechbrain'
        : 'pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu &amp;&amp; pip install deepfilternet speechbrain',
    });
  }
  if (health.ffmpeg === false) {
    notes.push({
      title: 'Video files cannot be read (ffmpeg is not installed)',
      body: 'Audio files work as normal. To pull the audio track out of mp4, mov, mkv and similar, install ffmpeg — the pip package below bundles it and needs no admin rights.',
      cmd: 'pip install imageio-ffmpeg',
    });
  }

  if (!notes.length) { box.classList.add('hidden'); box.innerHTML = ''; return; }
  box.classList.remove('hidden');
  box.innerHTML = notes.map((n) => `
    <div>
      <h3>${n.title}</h3>
      <p>${n.body}</p>
      <code>${n.cmd}</code>
    </div>`).join('');
}

async function loadMethodsAndPaths() {
  try {
    const [methods, paths] = await Promise.all([
      fetch(`${API}/api/methods`).then((r) => r.json()),
      fetch(`${API}/api/paths`).then((r) => r.json()),
    ]);
    state.methods = methods;
    state.paths = paths.paths || [];
    renderPaths();
    renderMethodBrowser();
    renderCustomRows();
  } catch (err) {
    $('#path-grid').innerHTML = '<div class="skeleton">could not reach the backend — is <code>python run.py serve</code> running?</div>';
  }
}

/* ------------------------------------------------------------------ paths */
/** Look a method up in the loaded method list (for the detail panels). */
function findMethod(kind, key) {
  const list = kind === 'denoise' ? state.methods.denoisers : state.methods.separators;
  return (list || []).find((m) => m.key === key) || null;
}

/** The "how this stage works" block: what it is, the algorithm, honest limits. */
function stageDetail(kind, key) {
  const m = findMethod(kind, key);
  if (!m) return '';
  return `
    <div class="stage-detail">
      <div class="sd-head">
        <span class="sd-kind">${kind === 'denoise' ? 'denoise' : 'separate'}</span>
        <span class="sd-name">${esc(m.name)}</span>
        ${m.latency ? `<span class="chip">${esc(m.latency)}</span>` : ''}
        <span class="chip ${m.available ? 'good' : 'warn'}">${m.available ? 'installed' : 'not installed'}</span>
      </div>
      ${m.how_it_works ? `<div class="detail-p">${esc(m.how_it_works)}</div>` : ''}
      ${(m.steps || []).length ? `
        <div>
          <h5 style="margin:0 0 4px;font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--text-faint)">Step by step</h5>
          <ol class="sd-steps">${m.steps.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>
        </div>` : ''}
      ${(m.strengths || []).length || (m.limitations || []).length ? `
        <div class="sd-lists">
          <div><h5>Good at</h5><ul class="good">${(m.strengths || []).map((s) => `<li>${esc(s)}</li>`).join('')}</ul></div>
          <div><h5>Watch out for</h5><ul class="bad">${(m.limitations || []).map((s) => `<li>${esc(s)}</li>`).join('')}</ul></div>
        </div>` : ''}
      ${m.reference ? `<div class="sd-ref">${esc(m.reference)}</div>` : ''}
    </div>`;
}

function renderPaths() {
  const grid = $('#path-grid');
  grid.innerHTML = '';
  state.paths.forEach((p) => {
    const card = el('div', `path-card${state.selected.has(p.id) ? ' selected' : ''}${p.available ? '' : ' blocked'}`);
    card.tabIndex = 0;
    card.setAttribute('role', 'checkbox');
    card.setAttribute('aria-checked', String(state.selected.has(p.id)));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); }
    });

    // One rolled-up reason + one command, instead of the same raw package
    // error repeated once per stage.
    const blocked = !p.available ? `
      <div class="pc-blocked">
        <div class="pb-why">Not ready: ${esc(p.blockers.join(' · '))}</div>
        ${p.install_hints.length ? `<div class="pb-cmd">${p.install_hints.map(esc).join('<br>')}</div>` : ''}
      </div>` : '';

    // The path runs, but not with the backend the preset names -- say which
    // stand-in took over, why, and what it would take to get the real one.
    const subs = (p.substitutions || []).length ? `
      <div class="pc-subs">
        ${p.substitutions.map((s) => `
          <div class="ps-line">Using <b>${esc(s.to_name)}</b> instead of ${esc(s.from_name)}</div>
          <div class="ps-why">${esc(s.reason)}</div>
          ${s.hint ? `<div class="pb-cmd">${esc(s.hint)}</div>` : ''}`).join('')}
      </div>` : '';

    card.innerHTML = `
      <span class="pc-check">✓</span>
      <div class="pc-top"><span class="pc-name">${esc(p.name)}</span></div>
      ${p.badge ? `<span class="pc-badge">${esc(p.badge)}</span>` : ''}
      <div class="pc-flow">${esc(p.denoiser_effective || p.denoiser)} → ${esc(p.separator_effective || p.separator)}</div>
      <div class="pc-desc">${esc(p.description)}</div>
      ${blocked}
      ${subs}
      <details class="pc-more">
        <summary>What this path does, in detail</summary>
        <div class="detail-body">
          ${p.detail ? `<div class="detail-p">${esc(p.detail)}</div>` : ''}
          ${p.best_for ? `<div class="detail-row good"><span class="dr-key">Best for</span><span class="dr-val">${esc(p.best_for)}</span></div>` : ''}
          ${p.avoid_when ? `<div class="detail-row warn"><span class="dr-key">Avoid when</span><span class="dr-val">${esc(p.avoid_when)}</span></div>` : ''}
          ${p.output ? `<div class="detail-row"><span class="dr-key">You get</span><span class="dr-val">${esc(p.output)}</span></div>` : ''}
          ${stageDetail('denoise', p.denoiser_effective || p.denoiser)}
          ${stageDetail('separate', p.separator_effective || p.separator)}
        </div>
      </details>
    `;

    // Clicking the card toggles selection, but not when the user is opening the
    // details panel or selecting text inside it.
    card.addEventListener('click', (event) => {
      if (event.target.closest('.pc-more')) return;
      if (!p.available) {
        toast(`"${p.name}" is not ready — ${p.blockers.join('; ')}`, 'warn', 9000);
        return;
      }
      if (state.selected.has(p.id)) state.selected.delete(p.id);
      else state.selected.add(p.id);
      card.classList.toggle('selected');
      card.setAttribute('aria-checked', String(state.selected.has(p.id)));
      persistSelection();
      updateRunState();
    });
    grid.appendChild(card);
  });
}

function renderMethodBrowser() {
  const render = (list, target) => {
    const box = $(target);
    box.innerHTML = '';
    list.forEach((m) => {
      const node = el('div', 'method');
      node.innerHTML = `
        <div class="method-top">
          <span class="method-name">${esc(m.name)}</span>
          <span class="pill ${m.available ? 'on' : 'off'}">${m.available ? 'ready' : 'not installed'}</span>
        </div>
        <div class="method-desc">${esc(m.description)}</div>
        <div class="chips">
          <span class="chip">${esc(m.family)}</span>
          <span class="chip">${esc(m.speed)}</span>
          <span class="chip dots">${'●'.repeat(m.quality)}${'○'.repeat(Math.max(0, 5 - m.quality))}</span>
          ${m.needs_gpu ? '<span class="chip warn">gpu advised</span>' : ''}
          ${m.offline ? '<span class="chip good">offline</span>' : '<span class="chip bad">uploads audio</span>'}
          ${m.max_speakers ? `<span class="chip">max ${m.max_speakers} spk</span>` : ''}
        </div>
        ${m.notes ? `<div class="method-desc">${esc(m.notes)}</div>` : ''}
        ${!m.available && m.install_hint ? `<div class="method-install">$ ${esc(m.install_hint)}</div>` : ''}
        ${m.how_it_works ? `
          <details class="pc-more">
            <summary>How ${esc(m.name)} works</summary>
            <div class="detail-body">${stageDetail(m.kind === 'denoise' ? 'denoise' : 'separate', m.key)}</div>
          </details>` : ''}
      `;
      box.appendChild(node);
    });
  };
  render(state.methods.denoisers || [], '#denoiser-list');
  render(state.methods.separators || [], '#separator-list');
}

/* ------------------------------------------------- custom path builder */
/* One row = one more path. You pick the denoising module (or "no denoising")
   and the separation module (or "no separation"); "+" appends another row, so
   several custom paths can be queued and run alongside the presets. */

function methodName(kind, key) {
  const m = findMethod(kind, key);
  return m ? m.name : key;
}

function methodAvailable(kind, key) {
  const m = findMethod(kind, key);
  return m ? Boolean(m.available) : false;
}

/** Display name for a custom row, e.g. "DeepFilterNet 3 + pyannote 3.1". */
function customName(row) {
  return `${methodName('denoise', row.denoiser)} + ${methodName('separate', row.separator)}`;
}

/** First installed method, preferring `preferred` when it is installed. */
function firstUsable(list, preferred) {
  const avail = (list || []).filter((m) => m.available);
  return ((avail.find((m) => m.key === preferred)) || avail[0] || { key: 'none' }).key;
}

function newRow() {
  return {
    id: `custom_${++state.rowSeq}`,
    denoiser: firstUsable(state.methods.denoisers, 'spectral_gate'),
    separator: firstUsable(state.methods.separators, 'diarize_cluster'),
  };
}

function addCustomRow() {
  const row = newRow();
  state.custom.push(row);
  state.selected.add(row.id);          // "+" means "I want to run this one"
  renderCustomRows();
  persistSelection();
  updateRunState();
}

function removeCustomRow(id) {
  state.custom = state.custom.filter((c) => c.id !== id);
  state.selected.delete(id);
  renderCustomRows();
  persistSelection();
  updateRunState();
}

/** Plain-English "what will this row actually do" line. */
function rowNote(row, index) {
  const dn = methodName('denoise', row.denoiser);
  const sn = methodName('separate', row.separator);
  const twin = state.custom.findIndex(
    (c, i) => i < index && c.denoiser === row.denoiser && c.separator === row.separator,
  );
  if (twin >= 0) return `Same combination as path ${twin + 1} — it would run twice.`;
  if (row.denoiser === 'none' && row.separator === 'none') {
    return 'Nothing happens: the file is copied through untouched.';
  }
  if (row.separator === 'none') return `${dn} cleans the audio and hands it back as one file.`;
  if (row.denoiser === 'none') return `No cleaning — ${sn} runs straight on the raw audio.`;
  return `${dn} cleans the audio, then ${sn} splits the speakers.`;
}

function blockersFor(row) {
  const out = [];
  [['denoise', row.denoiser], ['separate', row.separator]].forEach(([kind, key]) => {
    const m = findMethod(kind, key);
    if (m && !m.available) out.push(`${m.name}: ${m.unavailable_reason || 'not installed'}`);
  });
  return out;
}

function methodOptions(list, selectedKey) {
  return (list || []).map((m) => (
    `<option value="${esc(m.key)}"${m.key === selectedKey ? ' selected' : ''}` +
    `${m.available ? '' : ' disabled'}>${esc(m.name)}${m.available ? '' : '  (not installed)'}</option>`
  )).join('');
}

function renderCustomRows() {
  const box = $('#custom-rows');
  if (!box) return;
  box.innerHTML = '';

  state.custom.forEach((row, index) => {
    const blockers = blockersFor(row);
    if (blockers.length) state.selected.delete(row.id);
    const on = state.selected.has(row.id);

    const node = el('div', `crow${on ? ' on' : ''}${blockers.length ? ' blocked' : ''}`);
    node.innerHTML = `
      <input type="checkbox" class="crow-on" ${on ? 'checked' : ''} ${blockers.length ? 'disabled' : ''}
             aria-label="run path ${index + 1}" title="include this path in the run">
      <span class="crow-num">path${index + 1}</span>
      <label class="crow-field">
        <span>Denoising module</span>
        <select class="crow-den">${methodOptions(state.methods.denoisers, row.denoiser)}</select>
      </label>
      <span class="arrow">&rarr;</span>
      <label class="crow-field">
        <span>Separation module</span>
        <select class="crow-sep">${methodOptions(state.methods.separators, row.separator)}</select>
      </label>
      <button class="crow-del" type="button" aria-label="remove path ${index + 1}" title="remove this row">&#10005;</button>
      <div class="crow-note${blockers.length ? ' warn' : ''}">${
        blockers.length ? esc(`Not ready — ${blockers.join(' · ')}`) : esc(rowNote(row, index))
      }</div>
    `;

    node.querySelector('.crow-on').addEventListener('change', (ev) => {
      if (ev.target.checked) state.selected.add(row.id);
      else state.selected.delete(row.id);
      node.classList.toggle('on', ev.target.checked);
      persistSelection();
      updateRunState();
    });

    node.querySelector('.crow-den').addEventListener('change', (ev) => {
      row.denoiser = ev.target.value;
      renderCustomRows();
      persistSelection();
      updateRunState();
    });

    node.querySelector('.crow-sep').addEventListener('change', (ev) => {
      row.separator = ev.target.value;
      renderCustomRows();
      persistSelection();
      updateRunState();
    });

    node.querySelector('.crow-del').addEventListener('click', () => removeCustomRow(row.id));
    box.appendChild(node);
  });

  $('#custom-hint').innerHTML = state.custom.length
    ? 'Tick a row to run it. Greyed-out modules need a pip install &mdash; see the list below.'
    : 'No custom paths yet &mdash; press <b>+</b> to build one.';
}

function persistSelection() {
  localStorage.setItem('ds-selected', JSON.stringify([...state.selected]));
  localStorage.setItem('ds-custom', JSON.stringify(state.custom));
}

/** The builder always shows at least one row, so the feature is discoverable. */
function ensureFirstRow() {
  if (!state.custom.length) state.custom.push(newRow());
}

function restoreSelection() {
  try {
    const custom = JSON.parse(localStorage.getItem('ds-custom') || '[]');
    if (Array.isArray(custom)) {
      // keep only what a row actually needs; older builds stored a `name` too
      state.custom = custom
        .filter((c) => c && c.id && c.denoiser && c.separator)
        .map((c) => ({ id: String(c.id), denoiser: String(c.denoiser), separator: String(c.separator) }));
      state.custom.forEach((c) => {
        const n = /^custom_(\d+)$/.exec(c.id);
        if (n) state.rowSeq = Math.max(state.rowSeq, Number(n[1]));
      });
    }
    const sel = JSON.parse(localStorage.getItem('ds-selected') || '[]');
    if (Array.isArray(sel) && sel.length) {
      const valid = new Set([...state.paths.filter((p) => p.available).map((p) => p.id), ...state.custom.map((c) => c.id)]);
      const restored = sel.filter((id) => valid.has(id));
      if (restored.length) state.selected = new Set(restored);
    }
  } catch (err) { /* ignore corrupt storage */ }
  renderPaths();
  ensureFirstRow();
  renderCustomRows();
}

/* ----------------------------------------------------------------- upload */
function wireUpload() {
  const dz = $('#dropzone');
  const input = $('#file-input');

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => { if (input.files[0]) setFile(input.files[0]); });

  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.add('drag');
  }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.remove('drag');
  }));
  dz.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  const demoBtn = $('#load-demo');
  if (demoBtn) demoBtn.addEventListener('click', loadDemo);

  $('#clear-file').addEventListener('click', () => {
    state.file = null;
    input.value = '';
    $('#file-info').classList.add('hidden');
    $('#dropzone').classList.remove('hidden');
    updateRunState();
  });
}

/** True sample rate straight from the RIFF/WAVE header (null for other formats). */
async function wavHeaderRate(file) {
  try {
    const head = new DataView(await file.slice(0, 48).arrayBuffer());
    const tag = String.fromCharCode(head.getUint8(0), head.getUint8(1), head.getUint8(2), head.getUint8(3));
    if (tag !== 'RIFF') return null;
    const rate = head.getUint32(24, true);
    return rate > 1000 && rate < 400000 ? rate : null;
  } catch (err) {
    return null;
  }
}

async function loadDemo() {
  try {
    const res = await fetch('/static/assets/demo_conversation.wav');
    if (!res.ok) throw new Error('demo file not bundled — run: python run.py demo');
    const blob = await res.blob();
    await setFile(new File([blob], 'demo_conversation.wav', { type: 'audio/wav' }));
    toast('Loaded the 4-speaker demo (synthetic, 7 dB SNR).');
  } catch (err) {
    toast(String(err.message || err), 'error');
  }
}

function isVideoFile(file) {
  const exts = (state.health && state.health.video_formats) || [];
  const name = String(file.name || '').toLowerCase();
  return exts.some((ext) => name.endsWith(ext)) || String(file.type || '').startsWith('video/');
}

async function setFile(file) {
  state.file = file;
  $('#dropzone').classList.add('hidden');
  $('#file-info').classList.remove('hidden');
  $('#fi-name').textContent = file.name;
  $('#input-audio').src = URL.createObjectURL(file);

  const video = isVideoFile(file);
  if (video && state.health && state.health.ffmpeg === false) {
    toast('This is a video, but ffmpeg is not installed so the audio cannot be extracted. Run: pip install imageio-ffmpeg', 'error', 14000);
  }

  const chips = $('#fi-chips');
  chips.innerHTML = `<span class="chip">${fmtSize(file.size)}</span>`
    + (video ? '<span class="chip accent">video → audio track</span>' : '')
    + '<span class="chip">reading…</span>';

  const info = await localWaveform(file);
  if (info) {
    drawWave($('#input-wave'), info.peaks, '#5eead4');
    $('#input-wave').dataset.peaks = JSON.stringify(info.peaks);
    // decodeAudioData resamples to the device rate, so only trust a rate we
    // read straight out of the container header.
    const trueRate = await wavHeaderRate(file);
    chips.innerHTML =
      `<span class="chip">${fmtSize(file.size)}</span>` +
      (video ? '<span class="chip accent">video → audio track</span>' : '') +
      `<span class="chip">${fmtClock(info.duration)}</span>` +
      (trueRate ? `<span class="chip">${(trueRate / 1000).toFixed(1)} kHz</span>` : '') +
      `<span class="chip">${info.channels === 1 ? 'mono' : `${info.channels} ch`}</span>`;
  } else {
    chips.innerHTML = `<span class="chip">${fmtSize(file.size)}</span>`
      + (video ? '<span class="chip accent">video → audio track</span>' : '')
      + '<span class="chip warn">no preview in the browser — the backend still reads it</span>';
  }
  updateRunState();
}

/* ---------------------------------------------------------------- controls */
function wireControls() {
  $('#add-row').addEventListener('click', addCustomRow);
  $('#run-btn').addEventListener('click', startJob);
  window.addEventListener('resize', debounce(redrawAllWaves, 180));
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function redrawAllWaves() {
  $$('canvas.wave').forEach((c) => {
    const peaks = c.dataset.peaks ? JSON.parse(c.dataset.peaks) : null;
    if (peaks) drawWave(c, peaks, c.dataset.color || '#5eead4');
  });
}

function selectedSpecs() {
  const specs = [];
  state.selected.forEach((id) => {
    const custom = state.custom.find((c) => c.id === id);
    if (custom) specs.push({ id: custom.id, name: customName(custom), denoiser: custom.denoiser, separator: custom.separator });
    else specs.push({ id });
  });
  return specs;
}

function updateRunState() {
  const count = state.selected.size;
  const ready = Boolean(state.file) && count > 0;
  $('#run-btn').disabled = !ready || Boolean(state.polling);
  $('#run-summary').textContent = !state.file
    ? 'Choose a file and at least one path.'
    : count === 0
      ? 'Now tick at least one pipeline path.'
      : `${state.file.name} · ${count} path${count > 1 ? 's' : ''} selected`;
}

/* --------------------------------------------------------------- run a job */
async function startJob() {
  if (!state.file || !state.selected.size) return;

  const form = new FormData();
  form.append('file', state.file);
  form.append('paths', JSON.stringify(selectedSpecs()));
  form.append('options', JSON.stringify({
    num_speakers: $('#opt-speakers').value ? Number($('#opt-speakers').value) : null,
    count_on: $('#opt-count-on').value,
    normalize_tracks: $('#opt-normalize').checked,
  }));

  setRunning(true);
  $('#results').innerHTML = '';
  $('#card-compare').classList.add('hidden');
  $('#stage-list').innerHTML = '';
  $('#log').textContent = '';
  state.cards.clear();
  state.seq = 0;

  try {
    const res = await fetch(`${API}/api/jobs`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'upload rejected');
    }
    const data = await res.json();
    state.job = data.job_id;
    state.startedAt = Date.now();
    startTimer();
    pollLoop();
  } catch (err) {
    setRunning(false);
    toast(String(err.message || err), 'error', 9000);
  }
}

function setRunning(running) {
  state.polling = running ? true : null;
  $('#run-btn').disabled = running;
  $('#run-btn .btn-spinner').hidden = !running;
  $('#run-btn .btn-label').textContent = running ? 'Running…' : 'Run pipeline';

  // the progress card stays visible after the run so the log remains readable
  $('#card-progress').classList.remove('hidden');
  const badge = $('#card-progress .step');
  const title = $('#card-progress h2');
  if (running) {
    badge.className = 'step live';
    badge.textContent = '●';
    title.textContent = 'Working…';
  } else {
    badge.className = 'step';
    badge.textContent = '✓';
    title.textContent = 'Run finished';
    clearInterval(state.timer);
    state.timer = null;
  }
  updateRunState();
}

function startTimer() {
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    $('#elapsed').textContent = `${((Date.now() - state.startedAt) / 1000).toFixed(1)}s`;
  }, 100);
}

async function pollLoop() {
  if (!state.job) return;
  try {
    const res = await fetch(`${API}/api/jobs/${state.job}?since=${state.seq}`);
    if (!res.ok) throw new Error('job vanished');
    const data = await res.json();
    handleEvents(data);

    if (data.status === 'done') {
      finishJob(data.report);
      return;
    }
    if (data.status === 'error') {
      setRunning(false);
      showJobError(data.error, data.error_fix);
      return;
    }
  } catch (err) {
    setRunning(false);
    toast(`Lost contact with the backend: ${err.message}`, 'error');
    return;
  }
  setTimeout(pollLoop, POLL_MS);
}

function showJobError(message, fix) {
  toast(message || 'The job failed.', 'error', 14000);
  const box = el('section', 'card result-card failed');
  box.innerHTML = `
    <div class="card-head">
      <span class="step" style="background:var(--bad);color:#fff">!</span>
      <div><h2>That did not work</h2><p class="sub">Nothing was produced for this file.</p></div>
    </div>
    <div class="error-box">
      ${esc(message || 'The job failed.')}
      ${fix ? `<div class="error-fix"><b>Try this:</b> <code>${esc(fix)}</code></div>` : ''}
    </div>`;
  $('#results').innerHTML = '';
  $('#results').appendChild(box);
}

function handleEvents(data) {
  state.seq = data.next_seq || state.seq;
  $('#progress-fill').style.width = `${Math.min(100, (data.pct || 0) * 100)}%`;
  $('#progress-sub').textContent = data.message || '…';

  (data.events || []).forEach((ev) => {
    const line = `[${((ev.pct || 0) * 100).toFixed(0).padStart(3)}%] ${(ev.stage || '').padEnd(10)} ${ev.message || ''}`;
    const log = $('#log');
    log.textContent += `${line}\n`;
    log.parentElement.scrollTop = log.parentElement.scrollHeight;

    if (ev.path_id) updateStage(ev);
    if (ev.artifact) handleArtifact(ev);
  });
}

function updateStage(ev) {
  const id = `stage-${ev.path_id}`;
  let node = document.getElementById(id);
  if (!node) {
    node = el('div', 'stage running');
    node.id = id;
    node.innerHTML = `<span class="st-icon">◌</span><span class="st-name">${esc(pathLabel(ev.path_id))}</span><span class="st-msg"></span>`;
    $('#stage-list').appendChild(node);
  }
  node.querySelector('.st-msg').textContent = ev.message || '';
  if (ev.stage === 'path_done') {
    node.className = `stage ${ev.status === 'ok' ? 'done' : 'failed'}`;
    node.querySelector('.st-icon').textContent = ev.status === 'ok' ? '✓' : '✕';
  } else if (ev.stage === 'error') {
    node.className = 'stage failed';
    node.querySelector('.st-icon').textContent = '✕';
  }
}

function pathLabel(pathId) {
  const preset = state.paths.find((p) => p.id === pathId);
  if (preset) return preset.name;
  const custom = state.custom.find((c) => c.id === pathId);
  return custom ? customName(custom) : pathId;
}

/* -------------------------------------------- incremental result rendering */
function ensureCard(pathId) {
  if (state.cards.has(pathId)) return state.cards.get(pathId);
  const card = el('section', 'card result-card');
  card.innerHTML = `
    <div class="card-head">
      <span class="step">▶</span>
      <div>
        <h2 class="rc-title">${esc(pathLabel(pathId))}</h2>
        <p class="sub rc-flow" data-role="flow">working…</p>
      </div>
      <div class="head-right"><span class="chip" data-role="timing">running</span></div>
    </div>
    <div data-role="denoise"></div>
    <div data-role="speakers"></div>
    <div data-role="fun"></div>
  `;
  $('#results').appendChild(card);
  state.cards.set(pathId, card);
  return card;
}

function handleArtifact(ev) {
  const art = ev.artifact;
  const card = ensureCard(art.path_id);
  const url = `${API}/api/files/${state.job}/${art.path_id}/${art.file}`;

  if (art.kind === 'denoised') {
    const box = card.querySelector('[data-role="denoise"]');
    if (box.dataset.filled) return;
    box.dataset.filled = '1';
    box.innerHTML = `
      <div class="stage-block">
        <div class="sb-head"><span class="sb-num">1</span><span class="sb-title">Denoised audio</span>
          <span class="chip good">ready</span></div>
        <div class="sp-audio"><audio controls preload="metadata" src="${url}"></audio>
          <a class="ghost-btn" href="${url}" download="${esc(art.file)}">Download</a></div>
      </div>`;
  }

  if (art.kind === 'speaker') {
    const box = card.querySelector('[data-role="speakers"]');
    if (!box.dataset.filled) {
      box.dataset.filled = '1';
      box.innerHTML = `<div class="stage-block">
        <div class="sb-head"><span class="sb-num">2</span><span class="sb-title">Separated speakers</span>
        <span class="chip accent" data-role="spk-count">0 found</span></div>
        <div data-role="spk-rows"></div></div>`;
    }
    const rows = box.querySelector('[data-role="spk-rows"]');
    if (rows.querySelector(`[data-file="${art.file}"]`)) return;
    const color = SPEAKER_COLORS[(art.index || 0) % SPEAKER_COLORS.length];
    const row = el('div', 'speaker-row');
    row.dataset.file = art.file;
    row.innerHTML = `
      <div class="sp-id">
        <div class="sp-name"><span class="sp-swatch" style="background:${color}"></span>Speaker ${(art.index || 0) + 1}</div>
        <div class="sp-stats">measuring…</div>
      </div>
      <div class="sp-right">
        <div class="sp-audio"><audio controls preload="metadata" src="${url}"></audio>
          <a class="ghost-btn" href="${url}" download="${esc(art.file)}">Download</a></div>
      </div>`;
    rows.appendChild(row);
    box.querySelector('[data-role="spk-count"]').textContent = `${rows.children.length} found`;
  }
}

/* --------------------------------------------------------------- final view */
function finishJob(report) {
  setRunning(false);
  if (!report) { toast('job finished but no report came back', 'warn'); return; }

  $('#progress-fill').style.width = '100%';
  $('#progress-sub').textContent = `finished in ${report.elapsed_human}`;

  const est = report.speaker_estimate || {};
  if (est.n_speakers) {
    toast(`Detected ${est.n_speakers} speaker${est.n_speakers > 1 ? 's' : ''} · confidence ${Math.round((est.confidence || 0) * 100)}%`, 'info');
  }

  $('#results').innerHTML = '';
  state.cards.clear();
  (report.paths || []).forEach((p) => renderResultCard(p, report));
  renderComparison(report);
  redrawAllWaves();
  $('#card-compare').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderResultCard(path, report) {
  const card = el('section', `card result-card${path.status === 'ok' ? '' : ' failed'}`);
  const dur = (report.input && report.input.duration) || 1;

  const head = `
    <div class="card-head">
      <span class="step">${path.status === 'ok' ? '✓' : '✕'}</span>
      <div>
        <h2 class="rc-title">${esc(path.name)} ${path.badge ? `<span class="pc-badge">${esc(path.badge)}</span>` : ''}</h2>
        <p class="sub rc-flow">${esc(path.denoiser_name || path.denoiser)} → ${esc(path.separator_name || path.separator)}</p>
      </div>
      <div class="head-right rc-actions">
        <span class="chip">${fmtTime((path.timings || {}).total)}</span>
        ${(path.timings || {}).realtime_factor ? `<span class="chip">${num(path.timings.realtime_factor, 2)}× real time</span>` : ''}
        ${path.status === 'ok' ? `<a class="ghost-btn" href="${API}/api/jobs/${report.job_id}/zip?path_id=${encodeURIComponent(path.id)}" download="${esc(report.job_id)}_${esc(path.id)}.zip">Download this path (.zip)</a>` : ''}
      </div>
    </div>`;

  if (path.status !== 'ok') {
    card.innerHTML = `${head}
      <div class="error-box">
        <b>This path could not finish${path.failed_stage ? ' (' + esc(path.failed_stage) + ' stage)' : ''}.</b><br>
        ${esc(path.error || 'Unknown error.')}
        ${path.error_fix ? `<div class="error-fix"><b>Try this:</b> <code>${esc(path.error_fix)}</code></div>` : ''}
      </div>`;
    $('#results').appendChild(card);
    return;
  }

  const den = path.denoised || {};
  const dm = den.metrics || {};
  const sep = path.separation || {};
  const sm = sep.metrics || {};
  const fun = path.fun || {};
  const inputWave = (report.input || {}).waveform || [];

  const denoiseBlock = `
    <div class="stage-block">
      <div class="sb-head">
        <span class="sb-num">1</span><span class="sb-title">Denoising — ${esc(den.method_name || den.method)}</span>
        <span class="chip">${esc(den.backend || '')}</span>
        <span class="chip">${fmtTime(den.elapsed)}</span>
      </div>

      <div class="metric-grid">
        ${metric('SNR gain', num(dm.snr_improvement_db, 1, ' dB'), dm.snr_improvement_db > 3 ? 'good' : dm.snr_improvement_db > 0 ? '' : 'warn', `${num(dm.snr_before_db, 1)} → ${num(dm.snr_after_db, 1)} dB`)}
        ${metric('Noise floor', num(-Math.abs(dm.noise_reduction_db || 0), 1, ' dB'), 'good', 'lower is quieter')}
        ${metric('Speech kept', num((dm.speech_preserved || 0) * 100, 0, '%'), (dm.speech_preserved || 0) > 0.85 ? 'good' : (dm.speech_preserved || 0) > 0.7 ? 'warn' : 'bad', 'envelope correlation')}
        ${metric('Denoise score', num(dm.quality_score, 0, '/100'), (dm.quality_score || 0) > 70 ? 'good' : 'warn', 'blended proxy')}
      </div>

      <div class="ab">
        <div class="ab-side">
          <span class="ab-label">Before</span>
          <canvas class="wave" height="70" data-peaks='${JSON.stringify(inputWave)}' data-color="#64708a"></canvas>
          <audio controls preload="metadata" src="${API}/api/files/${report.job_id}/${encodeURIComponent((report.input.path || '').split(/[\\/]/).pop())}"></audio>
        </div>
        <div class="ab-side">
          <span class="ab-label">After</span>
          <canvas class="wave" height="70" data-peaks='${JSON.stringify(den.waveform || [])}' data-color="#5eead4"></canvas>
          <div class="sp-audio">
            <audio controls preload="metadata" src="${API}/api/files/${report.job_id}/${path.id}/${den.file}"></audio>
            <a class="ghost-btn" href="${API}/api/files/${report.job_id}/${path.id}/${den.file}" download="${esc(den.file)}">Save</a>
          </div>
        </div>
      </div>
    </div>`;

  const tracks = path.tracks || [];
  const totalTalk = tracks.reduce((acc, t) => acc + (t.total_speech || 0), 0) || 1;
  const speakersBlock = `
    <div class="stage-block">
      <div class="sb-head">
        <span class="sb-num">2</span><span class="sb-title">Speakers — ${esc(sep.method_name || sep.method)}</span>
        <span class="chip accent">${tracks.length} speaker${tracks.length === 1 ? '' : 's'}</span>
        <span class="chip">confidence ${num((sep.confidence || 0) * 100, 0, '%')}</span>
        <span class="chip">${fmtTime(sep.elapsed)}</span>
      </div>

      <div class="metric-grid">
        ${metric('Speakers found', tracks.length, 'good', sm.estimated_speakers ? `estimator said ${sm.estimated_speakers}` : '')}
        ${metric('Separation score', num(sm.separation_score, 0, '/100'), (sm.separation_score || 0) > 70 ? 'good' : 'warn', 'reference-free')}
        ${metric('Cross-talk', num(sm.mean_cross_correlation, 3), (sm.mean_cross_correlation || 0) < 0.2 ? 'good' : 'warn', 'lower = cleaner split')}
        ${metric('Energy kept', num((sm.energy_conservation || 0) * 100, 0, '%'), 'good', 'sum of tracks vs mixture')}
      </div>

      <div class="talkbar" title="share of speaking time">
        ${tracks.map((t, i) => {
          const pct = Math.max(3, 100 * (t.total_speech || 0) / totalTalk);
          return `<div style="width:${pct}%;background:${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}">${Math.round(pct)}%</div>`;
        }).join('')}
      </div>

      <div style="margin-top:14px">
        ${tracks.map((t, i) => speakerRow(t, i, report, path, dur)).join('')}
      </div>
    </div>`;

  const funBlock = `
    <div class="stage-block">
      <div class="sb-head"><span class="sb-num">★</span><span class="sb-title">Nice to know</span></div>
      <div class="fun-grid">
        ${funCard(fmtClock(fun.clip_seconds), 'clip length')}
        ${funCard(`${num(fun.silence_percent, 0, '%')}`, 'was silence')}
        ${funCard(`${fun.words_estimate || 0}`, 'words (rough, ~150 wpm)')}
        ${funCard(fun.most_talkative !== null && fun.most_talkative !== undefined ? `Speaker ${fun.most_talkative + 1}` : '—', 'talked the most')}
        ${funCard(`${num(fun.estimated_overlap_percent, 0, '%')}`, 'estimated overlap')}
        ${funCard(`${num(fun.megabytes_of_pcm, 1)} MB`, 'raw float32 audio')}
        ${funCard(`${num((path.timings || {}).realtime_factor, 2)}×`, 'of real time to process')}
        ${funCard(esc(sm.embedding || sep.backend || '—'), 'engine used')}
      </div>
    </div>`;

  card.innerHTML = head + denoiseBlock + speakersBlock + funBlock;
  $('#results').appendChild(card);
}

function speakerRow(track, index, report, path, duration) {
  const color = SPEAKER_COLORS[index % SPEAKER_COLORS.length];
  const url = `${API}/api/files/${report.job_id}/${path.id}/${track.file}`;
  const segments = (track.segments || []).map((s) => {
    const left = 100 * s[0] / duration;
    const width = Math.max(0.35, 100 * (s[1] - s[0]) / duration);
    return `<div class="tl-seg" style="left:${left}%;width:${width}%;background:${color}" title="${s[0].toFixed(1)}s – ${s[1].toFixed(1)}s"></div>`;
  }).join('');

  return `
    <div class="speaker-row">
      <div class="sp-id">
        <div class="sp-name"><span class="sp-swatch" style="background:${color}"></span>${esc(track.label)}</div>
        <div class="sp-stats">
          ${num(track.total_speech, 1, 's')} speech<br>
          ${(track.segments || []).length} turn${(track.segments || []).length === 1 ? '' : 's'}<br>
          ${num(track.energy_db, 1, ' dBFS')}
        </div>
      </div>
      <div class="sp-right">
        <div class="timeline">${segments}</div>
        <div class="tl-axis"><span>0:00</span><span>${fmtClock(duration)}</span></div>
        <div class="sp-audio">
          <audio controls preload="metadata" src="${url}"></audio>
          <a class="ghost-btn" href="${url}" download="${esc(track.file)}">Save</a>
        </div>
      </div>
    </div>`;
}

function metric(label, value, kind, note) {
  return `<div class="metric ${kind || ''}">
    <div class="m-label">${esc(label)}</div>
    <div class="m-value">${esc(value)}</div>
    ${note ? `<div class="m-note">${esc(note)}</div>` : ''}
  </div>`;
}

function funCard(value, label) {
  return `<div class="fun"><div class="f-value">${value}</div><div class="f-label">${esc(label)}</div></div>`;
}

/* -------------------------------------------------------------- comparison */
function renderComparison(report) {
  const cmp = report.comparison || {};
  const ok = (report.paths || []).filter((p) => p.status === 'ok');
  if (ok.length < 1) return;

  $('#card-compare').classList.remove('hidden');
  $('#compare-note').textContent = cmp.note || '';
  $('#download-zip').href = `${API}/api/jobs/${report.job_id}/zip`;
  $('#download-zip').setAttribute('download', `${report.job_id}.zip`);

  const rankBox = $('#ranking');
  rankBox.innerHTML = '';
  if ((cmp.ranking || []).length > 1) {
    const top = cmp.ranking[0].overall || 1;
    cmp.ranking.forEach((r, i) => {
      const row = el('div', 'rank-row');
      row.innerHTML = `
        <div class="rank-pos">${i + 1}</div>
        <div class="rank-bar-wrap">
          <div class="rank-name"><span>${esc(r.name)}</span>
            <small>denoise ${r.denoise_score} · separate ${r.separation_score} · ${r.n_speakers} spk · ${r.seconds}s</small></div>
          <div class="rank-bar"><div class="rank-fill" style="width:${Math.max(4, 100 * r.overall / (top || 1))}%"></div></div>
        </div>
        <div class="rank-score">${r.overall}</div>`;
      rankBox.appendChild(row);
    });
  }

  const table = $('#compare-table');
  const ids = ok.map((p) => p.id);
  const names = ok.map((p) => p.name);
  let html = `<thead><tr><th>Metric</th>${names.map((n) => `<th>${esc(n)}</th>`).join('')}</tr></thead><tbody>`;
  (cmp.rows || []).forEach((row) => {
    html += `<tr><td>${esc(row.label)}${row.unit ? ` <small style="opacity:.6">${esc(row.unit)}</small>` : ''}</td>`;
    ids.forEach((id) => {
      const v = row.values[id];
      const isBest = row.best === id && ids.length > 1;
      html += `<td class="${isBest ? 'best' : ''}">${v === null || v === undefined ? '—' : Number(v).toFixed(Math.abs(v) < 10 ? 2 : 1)}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody>';
  table.innerHTML = html;

  if (cmp.speaker_counts && !cmp.speaker_count_agreement) {
    toast('The paths disagree on how many speakers there are — check the counts column.', 'warn', 9000);
  }
}

document.addEventListener('DOMContentLoaded', boot);
