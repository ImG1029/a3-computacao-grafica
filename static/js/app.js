'use strict';

// ── State ─────────────────────────────────────────────────────────────────
const selection = {};   // { category: name | null }
let components  = {};   // { category: { label, options:[{name,url}] } }
let composeJob  = null; // debounce timer

// ── Bootstrap ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  await Promise.all([loadComponents(), refreshModelInfo()]);

  // Default: nenhum componente selecionado
  for (const cat of Object.keys(components)) {
    selection[cat] = null;
  }
  highlightAll();
  await compose();
});

// ── API helpers ───────────────────────────────────────────────────────────

async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

// ── Model info (header status) ────────────────────────────────────────────

async function refreshModelInfo() {
  try {
    const info = await (await fetch('/api/model-info')).json();
    const dot  = document.getElementById('dot');
    const txt  = document.getElementById('dot-text');
    if (info.trained) {
      dot.className = 'dot ok';
      txt.textContent = `Modelo ativo · ${info.subject_count} pessoas · ${info.image_count} fotos`;
    } else if (info.image_count > 0) {
      dot.className = 'dot warn';
      txt.textContent = `${info.subject_count} pessoas · Modelo não treinado`;
    } else {
      dot.className = 'dot warn';
      txt.textContent = 'Sem base de dados — execute download_lfw.py';
    }
  } catch { /* ignore */ }
}

// ── Components + sidebar ──────────────────────────────────────────────────

async function loadComponents() {
  const data = await (await fetch('/api/components')).json();
  components = data;
  buildSidebar();
}

const CAT_ICON = {
  cabelo:'💈', sobrancelhas:'〰️',
  olhos:'👁️', nariz:'👃', boca:'👄', queixo:'🫦',
};

function buildSidebar() {
  const panel = document.getElementById('sidebar-inner');
  panel.innerHTML = '';

  for (const [cat, data] of Object.entries(components)) {
    const sec = el('div', 'cat-section');

    const lbl = el('div', 'cat-label');
    lbl.textContent = `${CAT_ICON[cat] || ''} ${data.label}`;
    sec.appendChild(lbl);

    const grid = el('div', 'thumb-grid');
    grid.id = `grid-${cat}`;

    // "Nenhum" tile
    const none = el('div', 'thumb none-thumb');
    none.dataset.cat  = cat;
    none.dataset.name = '';
    none.innerHTML = '<span class="none-x">✕</span><span class="thumb-name">Nenhum</span>';
    none.addEventListener('click', () => pick(cat, null, none));
    grid.appendChild(none);

    for (const opt of data.options) {
      const tile = el('div', 'thumb');
      tile.dataset.cat  = cat;
      tile.dataset.name = opt.name;

      const img = document.createElement('img');
      img.src     = opt.url;
      img.alt     = opt.name;
      img.loading = 'lazy';

      const lbl2 = el('div', 'thumb-name');
      lbl2.textContent = opt.name;

      tile.appendChild(img);
      tile.appendChild(lbl2);
      tile.addEventListener('click', () => pick(cat, opt.name, tile));
      grid.appendChild(tile);
    }

    sec.appendChild(grid);
    panel.appendChild(sec);
  }
}

function pick(cat, name, tile) {
  selection[cat] = name || null;
  document.querySelectorAll(`#grid-${cat} .thumb`).forEach(t => t.classList.remove('active'));
  tile.classList.add('active');
  scheduleCompose();
}

function highlightAll() {
  for (const [cat, name] of Object.entries(selection)) {
    const grid = document.getElementById(`grid-${cat}`);
    if (!grid) continue;
    grid.querySelectorAll('.thumb').forEach(t => {
      t.classList.toggle('active', name ? t.dataset.name === name : t.dataset.name === '');
    });
  }
}

// ── Compose (debounced) ───────────────────────────────────────────────────

function scheduleCompose() {
  clearTimeout(composeJob);
  composeJob = setTimeout(compose, 70);
}

async function compose() {
  showOverlay(true);
  const body = {};
  for (const [cat, name] of Object.entries(selection)) if (name) body[cat] = name;
  try {
    const data = await post('/api/compose', body);
    document.getElementById('preview').src = data.image;
  } finally {
    showOverlay(false);
  }
}

function showOverlay(on) {
  document.getElementById('face-overlay').classList.toggle('show', on);
}

// ── Save ──────────────────────────────────────────────────────────────────

function savePortrait() {
  const src = document.getElementById('preview').src;
  if (!src || src === window.location.href) return;
  const a  = document.createElement('a');
  a.href   = src;
  a.download = 'retrato_falado_suspeito.png';
  a.click();
  setStatus('Retrato salvo!', 'ok');
}

// ── Recognize ─────────────────────────────────────────────────────────────

async function recognize() {
  const btn = document.getElementById('btn-rec');
  btn.disabled = true;
  setStatus('Reconhecendo...', '');
  closeResults();

  const body = {};
  for (const [cat, name] of Object.entries(selection)) if (name) body[cat] = name;

  try {
    const data = await post('/api/recognize', body);
    if (data.error) { setStatus(data.error, 'err'); return; }
    renderResults(data.matches);
    const best = data.matches[0];
    setStatus(`Melhor: ${best.display_name} — ${best.similarity}% similaridade`, 'ok');
  } catch {
    setStatus('Erro de comunicação com o servidor.', 'err');
  } finally {
    btn.disabled = false;
  }
}

function renderResults(matches) {
  const list = document.getElementById('results-list');
  list.innerHTML = '';

  for (const m of matches) {
    const gold = m.rank === 1;
    const card = el('div', `match-card${gold ? ' gold' : ''}`);

    const photoEl = m.face_image
      ? `<img class="match-photo" src="${m.face_image}" alt="${m.display_name}">`
      : `<div class="photo-placeholder">👤</div>`;

    card.innerHTML = `
      <div class="card-body">
        <div class="rank-badge">#${m.rank}</div>
        ${photoEl}
        <div class="match-meta">
          <div class="match-name">${m.display_name}</div>
          <div class="match-dist">Distância: ${m.distance}</div>
        </div>
        <div class="sim-pct">${m.similarity}%</div>
      </div>
      <div class="sim-track">
        <div class="sim-fill" style="width:0%"></div>
      </div>`;
    list.appendChild(card);

    // Animate bar after paint
    requestAnimationFrame(() => {
      card.querySelector('.sim-fill').style.width = `${Math.min(m.similarity, 100)}%`;
    });
  }

  document.getElementById('results-panel').classList.add('open');
}

function closeResults() {
  document.getElementById('results-panel').classList.remove('open');
}

// ── Train ─────────────────────────────────────────────────────────────────

async function trainModel() {
  const modal = document.getElementById('train-modal');
  const msg   = document.getElementById('train-msg');
  modal.classList.remove('hidden');
  msg.textContent = 'Iniciando treinamento...';

  try {
    const res = await post('/api/train', {});
    if (res.error) { modal.classList.add('hidden'); setStatus(res.error, 'err'); return; }
  } catch { modal.classList.add('hidden'); return; }

  const poll = setInterval(async () => {
    try {
      const s = await (await fetch('/api/train/status')).json();
      msg.textContent = s.message || s.status;

      if (s.status === 'done') {
        clearInterval(poll);
        modal.classList.add('hidden');
        setStatus('Modelo treinado com sucesso!', 'ok');
        await refreshModelInfo();
      } else if (s.status === 'error') {
        clearInterval(poll);
        modal.classList.add('hidden');
        setStatus(`Erro: ${s.message}`, 'err');
      }
    } catch { clearInterval(poll); modal.classList.add('hidden'); }
  }, 700);
}

// ── Utils ─────────────────────────────────────────────────────────────────

function setStatus(text, type) {
  const bar = document.getElementById('status-bar');
  bar.textContent = text;
  bar.className = `status-line${type ? ` ${type}` : ''}`;
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}
