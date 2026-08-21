const DATA = 'data/';

const SHEETS = {
  reverse: 'reverse.png',   // x_t   — the noisy state
  eps:     'eps_pred.png',  // eps^  — what the network outputs
  x0:      'x0hat.png'      // x0^   — implied clean image
};

const NOTES = {
  reverse: 'The state at step <em>t</em> — the same size as the finished image, never compressed.',
  eps:     'What the network actually outputs: an estimate of the <em>noise</em> present at step ' +
           '<em>t</em>. Not a picture of a digit.',
  x0:      'The network never draws this. It is computed from &epsilon;&#770; by rearranging the ' +
           'forward equation — blurry early, sharp late.'
};

const CLASS_COLOURS = ['#4E79A7','#F28E2B','#E15759','#76B7B2','#59A14F',
                       '#EDC948','#B07AA1','#FF9DA7','#9C755F','#8C8C8C'];

const $ = id => document.getElementById(id);
const loadJSON = f => fetch(DATA + f).then(r => {
  if (!r.ok) throw new Error(f + ' (' + r.status + ')');
  return r.json();
});
const loadImage = f => new Promise((res, rej) => {
  const im = new Image();
  im.onload = () => res(im);
  im.onerror = () => rej(new Error(f + ' failed to load'));
  im.src = DATA + f;
});
const optional = p => p.catch(() => null);

/* Draw one tile of a sprite sheet. `pad` is the gap baked in at export time —
   the manifold and diffusion sheets use 0, vae_recon and vae_interp use 1. */
function drawTile(ctx, sheet, col, row, tile, pad){
  const c = ctx.canvas, s = tile + (pad || 0);
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(sheet, col * s, row * s, tile, tile, 0, 0, c.width, c.height);
}

async function main(){
  const veil = $('loading');

  let manifoldMeta, manifoldImg, sched, revMeta, sheets,
      reconMeta, reconImg, scatter, dec, decImg;
  try {
    [manifoldMeta, manifoldImg, sched, revMeta, reconMeta, reconImg, scatter] =
      await Promise.all([
        loadJSON('vae_manifold.json'), loadImage('vae_manifold.png'),
        loadJSON('schedule.json'),     loadJSON('reverse_meta.json'),
        loadJSON('vae_recon.json'),    loadImage('vae_recon.png'),
        loadJSON('vae_scatter.json')
      ]);
    const keys = Object.keys(SHEETS);
    const imgs = await Promise.all(keys.map(k => loadImage(SHEETS[k])));
    sheets = Object.fromEntries(keys.map((k, i) => [k, imgs[i]]));
    // decoder stations are optional — the page still works before T23 is run
    dec    = await optional(loadJSON('layers_vae_decoder.json'));
    decImg = dec ? await optional(loadImage('layers_vae_decoder.png')) : null;
  } catch (err) {
    veil.classList.add('is-error');
    veil.innerHTML = '<span>Could not load the model outputs.<br>' +
      'Expected a <code>data/</code> folder beside this page.<br><br>' +
      '<small>' + err.message + '</small></span>';
    return;
  }

  const G        = manifoldMeta.grid;
  const [LO, HI] = manifoldMeta.range;
  const mTile    = manifoldMeta.tile;
  const T        = revMeta.T;
  const dTile    = revMeta.tile;
  const SEEDS    = revMeta.seeds;
  const abar     = sched.alpha_bar;

  /* ── shared state ───────────────────────────────────────── */
  let z    = [0.6, 0.6];    // the chosen draw, in latent coordinates
  let seed = 0;             // which precomputed diffusion trajectory
  let view = 'reverse';

  /* ── the start picker ───────────────────────────────────── */
  const pts = (scatter && scatter.points) || [];
  const labelled = !!(scatter && scatter.labelled) && pts.length && pts[0].length > 2;
  const map = $('startMap'), mctx = map.getContext('2d');
  const MW = map.width, MH = map.height;

  const toPx = (a, b) => [ (a - LO) / (HI - LO) * MW, (HI - b) / (HI - LO) * MH ];
  const toZ  = (x, y)  => [ LO + x / MW * (HI - LO), HI - y / MH * (HI - LO) ];

  function paintMap(){
    mctx.clearRect(0, 0, MW, MH);
    mctx.fillStyle = '#fff'; mctx.fillRect(0, 0, MW, MH);
    mctx.strokeStyle = '#EDF1F5'; mctx.lineWidth = 1;
    for (let v = Math.ceil(LO); v <= HI; v++){
      const [gx] = toPx(v, 0), [, gy] = toPx(0, v);
      mctx.beginPath(); mctx.moveTo(gx, 0); mctx.lineTo(gx, MH); mctx.stroke();
      mctx.beginPath(); mctx.moveTo(0, gy); mctx.lineTo(MW, gy); mctx.stroke();
    }
    for (const p of pts){
      const [x, y] = toPx(p[0], p[1]);
      mctx.fillStyle = labelled ? CLASS_COLOURS[p[2] % 10] : '#8FA0AF';
      mctx.globalAlpha = .5;
      mctx.fillRect(x - 1.4, y - 1.4, 2.8, 2.8);
    }
    mctx.globalAlpha = 1;
    const [sx, sy] = toPx(z[0], z[1]);
    mctx.strokeStyle = 'rgba(16,23,32,.35)'; mctx.lineWidth = 1;
    mctx.beginPath(); mctx.moveTo(sx, 0); mctx.lineTo(sx, MH);
    mctx.moveTo(0, sy); mctx.lineTo(MW, sy); mctx.stroke();
    mctx.beginPath(); mctx.arc(sx, sy, 8, 0, Math.PI * 2);
    mctx.fillStyle = 'rgba(255,255,255,.85)'; mctx.fill();
    mctx.strokeStyle = '#101720'; mctx.lineWidth = 2; mctx.stroke();
  }

  /* what actually lives near the chosen point — measured, not asserted */
  function neighbourhood(k = 40){
    if (!labelled) return null;
    const d = pts.map(p => [(p[0]-z[0])**2 + (p[1]-z[1])**2, p[2]])
                 .sort((a, b) => a[0] - b[0]).slice(0, k);
    const counts = {};
    for (const [, c] of d) counts[c] = (counts[c] || 0) + 1;
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return { digit: +top[0], share: top[1] / d.length, spread: Object.keys(counts).length };
  }

  /* ── Channel A: the decoder ─────────────────────────────── */
  const paStep = $('paStep');
  const decStations = dec ? dec.stations : null;
  if (decStations) paStep.max = decStations.length - 1;
  else { paStep.disabled = true; paStep.max = 0; }

  const gridIndex = () => {
    if (!dec) return 0;
    const N = dec.grid;
    const col = Math.min(N-1, Math.max(0, Math.round((z[0] - LO)/(HI - LO)*(N-1))));
    const row = Math.min(N-1, Math.max(0, Math.round((HI - z[1])/(HI - LO)*(N-1))));
    return row * N + col;
  };

  function paintChannelA(){
    const i = +paStep.value;
    if (!decStations || !decImg){
      // before T23 has been run: still show the final decoded image
      const col = Math.min(G-1, Math.max(0, Math.round((z[0]-LO)/(HI-LO)*(G-1))));
      const row = Math.min(G-1, Math.max(0, Math.round((HI-z[1])/(HI-LO)*(G-1))));
      drawTile($('paImage').getContext('2d'), manifoldImg, col, row, mTile, 0);
      $('paName').textContent  = 'image';
      $('paShape').textContent = '28×28×1';
      $('paCount').textContent = '784';
      $('paNote').innerHTML =
        'Run <code>T23_export_decoder_layers.py</code> to step through the decoder layers. ' +
        'The final image is shown meanwhile.';
      return;
    }
    const st = decStations[i], p = gridIndex();
    drawTile($('paImage').getContext('2d'), decImg, i, p, dec.tile, 0);
    $('paName').textContent  = st.name;
    $('paShape').textContent = st.shape.join('×');
    $('paCount').textContent = st.elements.toLocaleString('en-US');

    const row = $('paRow');
    row.innerHTML = '';
    decStations.forEach((s, k) => {
      const el = document.createElement('div');
      el.className = 'station' + (k === i ? ' is-current' : '');
      const cv = document.createElement('canvas');
      cv.width = cv.height = 44;
      drawTile(cv.getContext('2d'), decImg, k, p, dec.tile, 0);
      const holder = document.createElement('div');
      holder.className = 'station__tile'; holder.appendChild(cv);
      const num = document.createElement('div');
      num.className = 'station__num'; num.textContent = s.elements.toLocaleString('en-US');
      const nm = document.createElement('div');
      nm.className = 'station__name'; nm.textContent = s.name;
      el.append(holder, num, nm);
      el.addEventListener('click', () => { paStep.value = k; paintChannelA(); });
      row.appendChild(el);
    });
  }

  /* ── Channel B: the reverse process ─────────────────────── */
  const pbStep = $('pbStep');
  pbStep.max = T - 1;

  function paintSched(t){
    const c = $('pbSched'), g = c.getContext('2d');
    const W = c.width, H = c.height, pad = 14;
    g.clearRect(0, 0, W, H);
    g.strokeStyle = '#E3E8ED'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(pad, H - pad); g.lineTo(W - pad, H - pad); g.stroke();
    /* x runs noise -> image, matching the slider. Column i holds t = T-1-i, so
       plotting against i keeps the marker travelling with the control. */
    const px = i => pad + i / (T - 1) * (W - 2 * pad);
    const py = i => (H - pad) - abar[T - 1 - i] * (H - 2 * pad);
    g.strokeStyle = '#16808E'; g.lineWidth = 2;
    g.beginPath();
    for (let i = 0; i < T; i++) i ? g.lineTo(px(i), py(i)) : g.moveTo(px(i), py(i));
    g.stroke();
    const i = T - 1 - t, x = px(i), y = py(i);
    g.strokeStyle = 'rgba(22,128,142,.35)'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x, pad); g.lineTo(x, H - pad); g.stroke();
    g.fillStyle = '#16808E'; g.strokeStyle = '#fff'; g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, 5, 0, Math.PI * 2); g.fill(); g.stroke();
  }

  function paintChannelB(){
    const i = +pbStep.value, t = T - 1 - i;
    drawTile($('pbImage').getContext('2d'), sheets[view], i, seed, dTile, 0);
    $('pbT').textContent = 't = ' + t;
    paintSched(t);
    renderCompare();
  }

  /* ── the starting noise tile: column 0 of the reverse sheet is x_T ── */
  function paintStartNoise(){
    drawTile($('startNoise').getContext('2d'), sheets.reverse, 0, seed, dTile, 0);
  }

  /* ── I2: where the randomness lives (unchanged, approved) ── */
  const RECON_COLS = reconMeta.cols;
  let reconCol = 0;
  function renderCompare(){
    for (let r = 0; r < 5; r++){
      const cv = $('vr' + r);
      if (cv) drawTile(cv.getContext('2d'), reconImg, reconCol, r, mTile, 1);
    }
    const i = +pbStep.value;
    for (let s = 0; s < SEEDS; s++){
      const cv = $('ds' + s);
      if (cv) drawTile(cv.getContext('2d'), sheets[view], i, s, dTile, 0);
    }
    $('difStripT').textContent = 't = ' + (T - 1 - i);
  }
  const nextBtn = $('vaeNext');
  if (nextBtn) nextBtn.addEventListener('click', () => {
    reconCol = (reconCol + 1) % RECON_COLS;
    renderCompare();
  });

  /* ── one update for a new draw ──────────────────────────── */
  function setDraw(a, b){
    z = [Math.max(LO, Math.min(HI, a)), Math.max(LO, Math.min(HI, b))];
    $('startZ').textContent = 'z = (' + z[0].toFixed(2) + ', ' + z[1].toFixed(2) + ')';
    const n = neighbourhood();
    $('startVerdict').innerHTML = n
      ? 'Of the 40 encoded images nearest this point, <strong>' +
        Math.round(n.share * 100) + '% are ' + n.digit + 's</strong>' +
        (n.spread > 1 ? ' — ' + n.spread + ' different digits appear here.'
                      : ' — only one digit appears here.')
      : 'Re-export <code>vae_scatter.json</code> with labels to see which digits live here.';
    paintMap();
    paintChannelA();
    paintStartNoise();
  }

  map.addEventListener('click', e => {
    const r = map.getBoundingClientRect();
    const [a, b] = toZ((e.clientX - r.left) / r.width * MW,
                       (e.clientY - r.top) / r.height * MH);
    setDraw(a, b);
  });
  map.addEventListener('keydown', e => {
    const step = e.shiftKey ? 0.5 : 0.15;
    const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0],
                    ArrowUp: [0, step], ArrowDown: [0, -step] };
    if (!moves[e.key]) return;
    e.preventDefault();
    setDraw(z[0] + moves[e.key][0], z[1] + moves[e.key][1]);
  });

  /* Two independent draws, one button each.
     A single button re-rolled both at once, which implied the 2-D vector and
     the 32x32 noise field were the same sample — or at least the same digit
     class. They are unrelated, and one control taught that falsehood. */
  const gauss = () => {                              // Box-Muller
    const u = Math.random() || 1e-9, v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  $('newDrawVae').addEventListener('click', () => setDraw(gauss(), gauss()));

  $('newDrawDif').addEventListener('click', () => {
    seed = (seed + 1) % SEEDS;
    paintStartNoise();
    paintChannelB();
  });

  paStep.addEventListener('input', paintChannelA);
  pbStep.addEventListener('input', paintChannelB);

  /* I4 — select on [data-view], not .view: the class is shared for styling with
     the stations toggle, and selecting on it reached buttons with no data-view. */
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      view = btn.dataset.view;
      $('pbNote').innerHTML = NOTES[view];
      paintChannelB();
    });
  });

  if (labelled){
    $('startLegend').innerHTML = CLASS_COLOURS.map((c, i) =>
      '<span class="legend__item"><i style="background:' + c + '"></i>' + i + '</span>'
    ).join('');
  }

  setDraw(z[0], z[1]);
  paintChannelB();
  buildLayers();
  buildDims();
  buildElbo();
  veil.classList.add('is-gone');


  /* ── T20: changing the size of the latent space ──
     Four models identical but for latent_dim. Optional — the section hides
     itself if the export has not been run. */
  async function buildDims(){
    let meta, recon, prior;
    try {
      meta  = await loadJSON('latent_dims.json');
      recon = await loadImage('latent_dims_recon.png');
      prior = await loadImage('latent_dims_prior.png');
    } catch (e) {
      document.querySelector('.dims').style.display = 'none';
      return;
    }

    const N = meta.cols, TL = meta.tile, PAD = meta.pad || 0;
    let mode = 'recon';

    function fill(){
      const sheet = mode === 'recon' ? recon : prior;
      const wrap = $('dimsRows');
      wrap.innerHTML = '';

      const rows = mode === 'recon'
        ? [{ label: 'original', sub: 'the input', dim: null }].concat(
            meta.dims.map(d => ({ label: d.dim + (d.dim === 1 ? ' number' : ' numbers'),
                                  sub: d.compression + '× compression · loss ' + d.recon,
                                  dim: d.dim })))
        : meta.dims.map(d => ({ label: d.dim + (d.dim === 1 ? ' number' : ' numbers'),
                                sub: 'decoded from z ~ N(0, I)', dim: d.dim }));

      rows.forEach((r, ri) => {
        const el = document.createElement('div');
        el.className = 'dimrow' + (r.dim === 2 ? ' is-default' : '');
        const lab = document.createElement('div');
        lab.className = 'dimrow__label';
        lab.innerHTML = '<span class="dimrow__n">' + r.label + '</span>' +
                        '<span class="dimrow__sub">' + r.sub + '</span>' +
                        (r.dim === 2 ? '<span class="dimrow__tag">used by this page</span>' : '');
        const strip = document.createElement('div');
        strip.className = 'dimrow__strip';
        for (let c = 0; c < N; c++){
          const cv = document.createElement('canvas');
          cv.width = cv.height = 48;
          drawTile(cv.getContext('2d'), sheet, c, ri, TL, PAD);
          strip.appendChild(cv);
        }
        el.append(lab, strip);
        wrap.appendChild(el);
      });
    }

    document.querySelectorAll('[data-dims]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-dims]').forEach(b => b.classList.remove('is-active'));
        btn.classList.add('is-active');
        mode = btn.dataset.dims;
        fill();
      });
    });

    const lo = meta.dims[0], hi = meta.dims[meta.dims.length - 1];
    $('dimsVerdict').innerHTML =
      'More dimensions means better reconstruction and less compression — ' +
      'from <strong>' + lo.compression + '×</strong> at ' + lo.dim + ' to <strong>' +
      hi.compression + '×</strong> at ' + hi.dim + ', with reconstruction loss falling from ' +
      lo.recon + ' to ' + hi.recon + '. There is no correct answer, only a trade-off. ' +
      'Two dimensions is used everywhere else on this page for one reason: it is the largest ' +
      'latent space that can be <strong>drawn</strong>.';

    fill();
  }

  /* ── ELBO: the two loss terms, over training ─────────────
     Reconstruction and KL live on very different scales (hundreds vs
     single digits), so a shared axis would flatten KL to a line near
     zero. Each gets its own axis, coloured to match its curve. */
  async function buildElbo(){
    let losses;
    try { losses = await loadJSON('vae_losses.json'); }
    catch (e) { document.querySelector('.elbo').style.display = 'none'; return; }

    const c = $('elboChart'), g = c.getContext('2d');
    const W = c.width, H = c.height, padL = 42, padR = 42, padT = 16, padB = 26;
    const n = losses.length;
    const recon = losses.map(r => r.recon), kl = losses.map(r => r.kl);
    const rMin = Math.min(...recon), rMax = Math.max(...recon);
    const kMin = Math.min(...kl), kMax = Math.max(...kl);
    const RECON_C = '#B4436C', KL_C = '#16808E';

    const px  = i => padL + (n === 1 ? 0 : i / (n - 1) * (W - padL - padR));
    const pyR = v => (H - padB) - (rMax > rMin ? (v - rMin) / (rMax - rMin) : .5) * (H - padT - padB);
    const pyK = v => (H - padB) - (kMax > kMin ? (v - kMin) / (kMax - kMin) : .5) * (H - padT - padB);

    g.clearRect(0, 0, W, H);
    g.strokeStyle = '#E3E8ED'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, H - padB); g.lineTo(W - padR, H - padB); g.stroke();

    function line(vals, py, color){
      g.strokeStyle = color; g.lineWidth = 2;
      g.beginPath();
      vals.forEach((v, i) => i ? g.lineTo(px(i), py(v)) : g.moveTo(px(i), py(v)));
      g.stroke();
      g.fillStyle = color;
      vals.forEach((v, i) => { g.beginPath(); g.arc(px(i), py(v), 3, 0, Math.PI * 2); g.fill(); });
    }
    line(recon, pyR, RECON_C);
    line(kl, pyK, KL_C);

    g.font = '11px ui-monospace, Consolas, monospace';
    g.fillStyle = RECON_C; g.textAlign = 'left';
    g.fillText(rMax.toFixed(0), 4, pyR(rMax) + 4);
    g.fillText(rMin.toFixed(0), 4, pyR(rMin) + 4);
    g.fillStyle = KL_C; g.textAlign = 'right';
    g.fillText(kMax.toFixed(1), W - 4, pyK(kMax) + 4);
    g.fillText(kMin.toFixed(1), W - 4, pyK(kMin) + 4);
    g.fillStyle = '#939EA9'; g.textAlign = 'center';
    losses.forEach((r, i) => g.fillText('epoch ' + r.epoch, px(i), H - 8));

    const first = losses[0], last = losses[n - 1];
    const reconDrop = Math.round((1 - last.recon / first.recon) * 100);
    const klRise = Math.round((last.kl / first.kl - 1) * 100);
    $('verdictElbo').innerHTML =
      'Over ' + n + ' epochs, reconstruction loss fell <strong>' + reconDrop + '%</strong> (' +
      first.recon.toFixed(0) + ' → ' + last.recon.toFixed(0) + '), while KL divergence ' +
      '<strong>rose ' + klRise + '%</strong> (' + first.kl.toFixed(1) + ' → ' + last.kl.toFixed(1) +
      '). They move in opposite directions because reconstruction dominates the total by roughly ' +
      Math.round(first.recon / first.kl) + '×: pushing it down means encoding more about each ' +
      'image, which is exactly what pulls the encoder\'s distribution away from the standard ' +
      'normal it is penalised for leaving.';
  }

  /* ── I5: how the input is compressed (unchanged) ────────── */
  async function buildLayers(){
    let lv, lvImg, ld, ldImg;
    try {
      [lv, lvImg, ld, ldImg] = await Promise.all([
        loadJSON('layers_vae.json'), loadImage('layers_vae.png'),
        loadJSON('layers_diffusion.json'), loadImage('layers_diffusion.png')
      ]);
    } catch (e) {
      document.querySelector('.layers').style.display = 'none';
      return;
    }

    const all = lv.stations.concat(ld.stations);
    const lgMin = Math.log10(Math.min(...all.map(s => s.elements)));
    const lgMax = Math.log10(Math.max(...all.map(s => s.elements)));
    const barH = n => 5 + (Math.log10(n) - lgMin) / (lgMax - lgMin) * 73;
    const maxSide = Math.max(...all.filter(s => s.spatial).map(s => s.shape[0]));
    const tileSize = s => s.spatial ? Math.round(26 + 38 * (s.shape[0] / maxSide)) : 30;
    const fmt = n => n.toLocaleString('en-US');
    const shapeLabel = s => s.shape.join('×');

    function card(st, sheet, idx, tile){
      const el = document.createElement('div');
      const side = (st.branch || 'main') !== 'main';
      el.className = 'station' + (st.spatial ? '' : ' station--vec') +
                     (side ? ' station--side' : '');
      el.setAttribute('aria-label',
        st.name + ', shape ' + shapeLabel(st) + ', ' + fmt(st.elements) + ' values');
      const cv = document.createElement('canvas');
      cv.width = cv.height = tileSize(st);
      drawTile(cv.getContext('2d'), sheet, idx, 0, tile, 0);
      const holder = document.createElement('div');
      holder.className = 'station__tile'; holder.style.height = '68px';
      holder.appendChild(cv);
      const bar = document.createElement('div');
      bar.className = 'station__bar';
      const fill = document.createElement('span');
      fill.style.height = barH(st.elements).toFixed(1) + 'px';
      bar.appendChild(fill);
      const shape = document.createElement('div');
      shape.className = 'station__shape'; shape.textContent = shapeLabel(st);
      const num = document.createElement('div');
      num.className = 'station__num'; num.textContent = fmt(st.elements);
      const nm = document.createElement('div');
      nm.className = 'station__name';
      nm.textContent = side ? st.name + ' · t' : st.name;
      el.append(holder, bar, num, shape, nm);
      return el;
    }

    function fillRow(rowId, meta, sheet, filter){
      const row = $(rowId); row.innerHTML = '';
      meta.stations.forEach((st, i) => {
        if (filter && !filter(st, i)) return;
        row.appendChild(card(st, sheet, i, meta.tile));
      });
    }

    /* Only main-path stations count. The U-Net's timestep embedding is a side
       branch: its 64 values are not a bottleneck the image passes through.
       Filtering on `spatial` instead would break the VAE, whose bottleneck IS
       a dense layer. */
    function summary(dlId, stations){
      const main = stations.filter(s => (s.branch || 'main') === 'main');
      const inp = main[0].elements;
      const peak = Math.max(...main.map(s => s.elements));
      const inner = main.slice(1, -1);
      const floor = inner.length ? Math.min(...inner.map(s => s.elements))
                                 : main[main.length - 1].elements;
      const out = main[main.length - 1].elements;
      $(dlId).innerHTML = [
        ['input', fmt(inp)],
        ['peak', fmt(peak) + '  (' + (peak/inp).toFixed(1) + '×)'],
        ['narrowest inside', fmt(floor) + '  (' + (floor/inp).toFixed(1) + '×)'],
        ['output', fmt(out)]
      ].map(([k, v]) => '<div><dt>' + k + '</dt><dd class="num">' + v + '</dd></div>').join('');
    }

    fillRow('lrowVae', lv, lvImg, null);
    summary('lsumVae', lv.stations);
    const keyOnly = st => st.key;
    fillRow('lrowDif', ld, ldImg, keyOnly);
    summary('lsumDif', ld.stations.filter(keyOnly));

    document.querySelectorAll('[data-stations]').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('[data-stations]').forEach(b => b.classList.remove('is-active'));
        btn.classList.add('is-active');
        const showAll = btn.dataset.stations === 'all';
        fillRow('lrowDif', ld, ldImg, showAll ? null : keyOnly);
        summary('lsumDif', showAll ? ld.stations : ld.stations.filter(keyOnly));
        $('lrowDif').classList.toggle('lrow--dense', showAll);
      });
    });

    const vIn = lv.stations[0].elements, vOut = lv.stations[lv.stations.length-1].elements;
    const dMain = ld.stations.filter(s => (s.branch || 'main') === 'main');
    const dIn = dMain[0].elements;
    const dFloor = Math.min(...dMain.slice(1, -1).map(s => s.elements));
    $('verdictLayers').innerHTML =
      'Both networks expand before they contract. Only one ever gets below its input: the ' +
      'encoder ends at <strong>' + vOut + '</strong> numbers, a ' + Math.round(vIn/vOut) +
      '× reduction. The U-Net\'s narrowest interior layer still holds <strong>' + fmt(dFloor) +
      '</strong> — ' + (dFloor/dIn).toFixed(0) + '× the image it was given — and it ends back at ' +
      'full size. It never forms an embedding at all.';
  }
}

main();