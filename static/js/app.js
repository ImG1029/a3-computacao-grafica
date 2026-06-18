'use strict';

const selection = {};   // { category: name | null }
let components  = {};   // { category: { label, options:[{name,url,group}] } }
let composeJob  = null; // debounce timer

// Compatibilidade: só é possível misturar componentes de faces do mesmo grupo
// (pose/proporção). Ver compute_groups.py / faces/groups.json.
const faceGroups  = {}; // { name: groupId }
let   groupLabels = {}; // { groupId: label }
let   groupingOn  = false;

document.addEventListener('DOMContentLoaded', async () => {
  await loadComponents();

  // Default: nenhum componente selecionado
  for (const cat of Object.keys(components)) {
    selection[cat] = null;
  }
  highlightAll();
  await compose();
});

async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function loadComponents() {
  const data = await (await fetch('/api/components')).json();
  // Compat: API antiga devolvia as categorias na raiz; a nova usa {categories, groups}.
  components  = data.categories || data;
  groupLabels = data.groups || {};
  groupingOn  = Object.keys(groupLabels).length > 0;

  for (const cat of Object.values(components)) {
    for (const opt of cat.options) {
      if (opt.group !== undefined && opt.group !== null) faceGroups[opt.name] = opt.group;
    }
  }
  buildSidebar();
}

const CAT_ICON = {
  cabelo:'💈', sobrancelhas:'〰️',
  olhos:'👁️', nariz:'👃', boca:'👄', queixo:'🫦',
};

function buildSidebar() {
  const panel = document.getElementById('sidebar-inner');
  panel.innerHTML = '';

  if (groupingOn) {
    const banner = el('div', 'group-banner');
    banner.id = 'group-banner';
    banner.innerHTML =
      '<span id="group-banner-text"></span>' +
      '<button class="group-reset" id="group-reset" onclick="switchGroup()">Trocar grupo</button>';
    panel.appendChild(banner);
  }

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
      if (opt.group !== undefined && opt.group !== null) tile.dataset.group = opt.group;

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
  if (tile.classList.contains('disabled')) return;   // face de grupo incompatível
  selection[cat] = name || null;
  document.querySelectorAll(`#grid-${cat} .thumb`).forEach(t => t.classList.remove('active'));
  tile.classList.add('active');
  applyGroupFilter();
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
  applyGroupFilter();
}

/** Grupo definido pelos componentes já selecionados (null = nenhum, livre). */
function activeGroup() {
  for (const name of Object.values(selection)) {
    if (name && name in faceGroups) return faceGroups[name];
  }
  return null;
}

/** Desabilita as faces de grupos incompatíveis e atualiza o banner. */
function applyGroupFilter() {
  if (!groupingOn) return;
  const ag = activeGroup();

  document.querySelectorAll('.thumb').forEach(t => {
    if (t.classList.contains('none-thumb')) return;          // "Nenhum" sempre liberado
    const g = t.dataset.group;
    const incompatible = ag !== null && g !== undefined && Number(g) !== ag;
    t.classList.toggle('disabled', incompatible);
    t.title = incompatible
      ? `Bloqueado: ${groupLabels[g] ?? 'outro grupo'} é incompatível com ${groupLabels[ag]}`
      : '';
  });

  const banner = document.getElementById('group-banner-text');
  const reset  = document.getElementById('group-reset');
  if (banner) {
    if (ag === null) {
      banner.textContent = 'Escolha um componente para fixar o grupo do rosto.';
      if (reset) reset.style.visibility = 'hidden';
    } else {
      banner.textContent = `Grupo ativo: ${groupLabels[ag]} — só faces compatíveis.`;
      if (reset) reset.style.visibility = 'visible';
    }
  }
}

/** Limpa a seleção para permitir começar de outro grupo. */
function switchGroup() {
  clearSelection();
}

/** Remove todos os componentes da montagem e devolve o retrato em branco. */
function clearSelection() {
  for (const cat of Object.keys(selection)) selection[cat] = null;
  highlightAll();          // marca os tiles "Nenhum" e reaplica o filtro de grupo
  scheduleCompose();       // seleção vazia → backend devolve canvas branco
  setStatus('Composição limpa.', '');
}

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

function savePortrait() {
  const src = document.getElementById('preview').src;
  if (!src || src === window.location.href) return;
  const a  = document.createElement('a');
  a.href   = src;
  a.download = 'retrato_falado_suspeito.png';
  a.click();
  setStatus('Retrato salvo!', 'ok');
}

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
