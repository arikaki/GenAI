/* ============================================================
   Two ways to make an image — I1 (shared scrubber) + I4 (view toggle)

   Reads the precomputed exports in data/. Nothing is inferred about
   sprite layout: tile size, grid size and column counts all come from
   the accompanying JSON, so re-exporting with different settings does
   not break the page.
   ============================================================ */

const DATA = 'data/';

/* sprite sheets are laid out rows = seed, cols = timestep (t = T-1 … 0) */
const SHEETS = {
  reverse: 'reverse.png',   // x_t   — the noisy state
  eps:     'eps_pred.png',  // eps^  — what the network outputs
  x0:      'x0hat.png'      // x0^   — implied clean image
};

/* the sentence under channel B changes with the view: this is the C2 payload */
const NOTES = {
  reverse: 'The state at step <em>t</em> — the same size as the finished image, never compressed.',
  eps:     'What the network actually outputs: an estimate of the <em>noise</em> present at step ' +
           '<em>t</em>. Not a picture of a digit.',
  x0:      'The network never draws this. It is computed from &epsilon;&#770; by rearranging the ' +
           'forward equation — blurry early, sharp late.'
};

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

/* ── latent path ────────────────────────────────────────────
   The scrubber walks a smooth loop through the 2-D latent plane.
   A rosette rather than a circle so it visits more of the space. */
function latentAt(frac){
  const th = frac * Math.PI * 2;
  const r  = 1.55 + 0.75 * Math.sin(3 * th);
  return [r * Math.cos(th), r * Math.sin(th)];
}

/* Draw one tile of a sprite sheet, filling the destination canvas.
   `pad` is the gap baked between tiles at export time — the manifold and the
   diffusion sheets use 0, vae_recon and vae_interp use 1. Getting this wrong
   shifts every tile by a pixel per column, which compounds across 16 columns. */
function drawTile(ctx, sheet, col, row, tile, pad){
  const c = ctx.canvas, s = tile + (pad || 0);
  ctx.imageSmoothingEnabled = false;      // keep the pixel grid honest
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(sheet, col * s, row * s, tile, tile, 0, 0, c.width, c.height);
}

async function main(){
  const veil = $('loading');

  let manifoldMeta, manifoldImg, sched, revMeta, sheets, reconMeta, reconImg;
  try {
    [manifoldMeta, manifoldImg, sched, revMeta, reconMeta, reconImg] = await Promise.all([
      loadJSON('vae_manifold.json'),
      loadImage('vae_manifold.png'),
      loadJSON('schedule.json'),
      loadJSON('reverse_meta.json'),
      loadJSON('vae_recon.json'),
      loadImage('vae_recon.png')
    ]);
    const keys = Object.keys(SHEETS);
    const imgs = await Promise.all(keys.map(k => loadImage(SHEETS[k])));
    sheets = Object.fromEntries(keys.map((k, i) => [k, imgs[i]]));
  } catch (err) {
    veil.classList.add('is-error');
    veil.innerHTML = '<span>Could not load the model outputs.<br>' +
      'Expected a <code>data/</code> folder beside this page.<br><br>' +
      '<small>' + err.message + '</small></span>';
    return;
  }

  /* everything below is driven by these numbers, not by assumptions */
  const G        = manifoldMeta.grid;              // 30
  const [LO, HI] = manifoldMeta.range;             // [-3, 3]
  const mTile    = manifoldMeta.tile;              // 28
  const T        = revMeta.T;                      // 200
  const dTile    = revMeta.tile;                   // 32
  const abar     = sched.alpha_bar;
  const dims     = dTile * dTile;

  const vaeImg  = $('vaeImage').getContext('2d');
  const vaeMap  = $('vaeMap').getContext('2d');
  const difImg  = $('difImage').getContext('2d');
  const difSch  = $('difSched').getContext('2d');
  const scrub   = $('scrub');

  scrub.max = T - 1;
  $('difDim').textContent = dims;

  let view = 'reverse';

  /* ── latent map: the manifold itself is the map ── */
  function paintMap(z1, z2){
    const c = vaeMap.canvas, W = c.width, H = c.height;
    vaeMap.imageSmoothingEnabled = true;
    vaeMap.clearRect(0, 0, W, H);
    vaeMap.globalAlpha = .34;                       // recede: it is a map, not the specimen
    vaeMap.drawImage(manifoldImg, 0, 0, W, H);
    vaeMap.globalAlpha = 1;

    const x = (z1 - LO) / (HI - LO) * W;
    const y = (HI - z2) / (HI - LO) * H;

    vaeMap.strokeStyle = 'rgba(180,67,108,.30)';    // crosshair
    vaeMap.lineWidth = 1;
    vaeMap.beginPath();
    vaeMap.moveTo(x, 0); vaeMap.lineTo(x, H);
    vaeMap.moveTo(0, y); vaeMap.lineTo(W, y);
    vaeMap.stroke();

    vaeMap.fillStyle = '#B4436C';
    vaeMap.strokeStyle = '#fff';
    vaeMap.lineWidth = 2;
    vaeMap.beginPath();
    vaeMap.arc(x, y, 5, 0, Math.PI * 2);
    vaeMap.fill(); vaeMap.stroke();
  }

  /* ── schedule curve: signal remaining, alpha-bar ──
     x runs noise → image, matching the scrubber. Column i of the sprite
     holds t = T-1-i, so plotting against i keeps both moving together.
     If this ran on t instead, dragging right would move the marker left. */
  function paintSched(t){
    const c = difSch.canvas, W = c.width, H = c.height, pad = 14;
    difSch.clearRect(0, 0, W, H);

    difSch.strokeStyle = '#E3E8ED'; difSch.lineWidth = 1;
    difSch.beginPath();
    difSch.moveTo(pad, H - pad); difSch.lineTo(W - pad, H - pad);
    difSch.stroke();

    const px = i => pad + i / (T - 1) * (W - 2 * pad);          // i = scrubber position
    const py = i => (H - pad) - abar[T - 1 - i] * (H - 2 * pad); // abar at t = T-1-i

    difSch.strokeStyle = '#16808E'; difSch.lineWidth = 2;
    difSch.beginPath();
    for (let i = 0; i < T; i++) i ? difSch.lineTo(px(i), py(i)) : difSch.moveTo(px(i), py(i));
    difSch.stroke();

    const i = T - 1 - t, x = px(i), y = py(i);

    difSch.strokeStyle = 'rgba(22,128,142,.35)';
    difSch.lineWidth = 1;
    difSch.beginPath();
    difSch.moveTo(x, pad); difSch.lineTo(x, H - pad);
    difSch.stroke();

    difSch.fillStyle = '#16808E'; difSch.strokeStyle = '#fff'; difSch.lineWidth = 2;
    difSch.beginPath(); difSch.arc(x, y, 5, 0, Math.PI * 2);
    difSch.fill(); difSch.stroke();
  }

  /* ── calibrated tick marks under the control ── */
  function paintTicks(){
    const c = $('ticks'), g = c.getContext('2d');
    g.clearRect(0, 0, c.width, c.height);
    for (let i = 0; i <= 20; i++){
      const x = i / 20 * (c.width - 2) + 1;
      const major = i % 5 === 0;
      g.strokeStyle = major ? '#8A97A3' : '#4A555F';
      g.lineWidth = major ? 1.5 : 1;
      g.beginPath();
      g.moveTo(x, c.height);
      g.lineTo(x, c.height - (major ? 11 : 6));
      g.stroke();
    }
  }

  /* ── I2: where the randomness lives ──
     Channel A — one input image, encoded five times: the encoder emits a
     distribution, so each draw lands on a slightly different z.
     Channel B — three reverse runs that differ only in their starting noise. */
  const RECON_COLS = reconMeta.cols;      // 16 test images
  let reconCol = 0;

  function renderCompare(){
    for (let r = 0; r < 5; r++){
      const cv = $('vr' + r);
      if (cv) drawTile(cv.getContext('2d'), reconImg, reconCol, r, mTile, 1);
    }
    const i = +scrub.value;
    for (let s = 0; s < revMeta.seeds; s++){
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

  /* ── I5: how the input is compressed ──
     Built from the real exported station lists. Two encodings, because the data
     tells two stories: tile size follows the spatial dimension (the U-Net
     hourglass), bar height follows element count on a log scale shared across
     both models (so 2 and 4096 are directly comparable). */
  async function buildLayers(){
    let lv, lvImg, ld, ldImg;
    try {
      [lv, lvImg, ld, ldImg] = await Promise.all([
        loadJSON('layers_vae.json'), loadImage('layers_vae.png'),
        loadJSON('layers_diffusion.json'), loadImage('layers_diffusion.png')
      ]);
    } catch (e) {
      document.querySelector('.layers').style.display = 'none';   // section is optional
      return;
    }

    const all = lv.stations.concat(ld.stations);
    const lgMin = Math.log10(Math.min(...all.map(s => s.elements)));
    const lgMax = Math.log10(Math.max(...all.map(s => s.elements)));
    const BAR_MIN = 5, BAR_MAX = 78;
    const barH = n => BAR_MIN +
      (Math.log10(n) - lgMin) / (lgMax - lgMin) * (BAR_MAX - BAR_MIN);

    const maxSide = Math.max(...all.filter(s => s.spatial).map(s => s.shape[0]));
    const tileSize = s => s.spatial
      ? Math.round(26 + 38 * (s.shape[0] / maxSide))
      : 30;

    const fmt = n => n.toLocaleString('en-US');
    const shapeLabel = s => s.shape.length === 3
      ? s.shape[0] + '×' + s.shape[1] + '×' + s.shape[2]
      : s.shape.join('×');

    function card(st, sheet, idx, tile){
      const el = document.createElement('div');
      const side = (st.branch || 'main') !== 'main';
      el.className = 'station' + (st.spatial ? '' : ' station--vec') +
                     (side ? ' station--side' : '');
      el.setAttribute('aria-label',
        st.name + ', shape ' + shapeLabel(st) + ', ' + fmt(st.elements) + ' values');

      const px = tileSize(st);
      const cv = document.createElement('canvas');
      cv.width = cv.height = px;
      drawTile(cv.getContext('2d'), sheet, idx, 0, tile, 0);

      const holder = document.createElement('div');
      holder.className = 'station__tile';
      holder.style.height = (Math.round(26 + 38) + 4) + 'px';
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

    /* summary strip — the numbers that carry the argument.
       Only main-path stations count. The U-Net's timestep embedding is a side
       branch: its 64 values are not a bottleneck the image passes through, and
       including it would report 64 instead of 4096. Filtering on `spatial`
       instead would break the VAE, whose bottleneck is a dense layer. */
    function summary(dlId, stations){
      const main = stations.filter(s => (s.branch || 'main') === 'main');
      const inp = main[0].elements;
      const peak = Math.max(...main.map(s => s.elements));
      const inner = main.slice(1, -1);
      const floor = inner.length
        ? Math.min(...inner.map(s => s.elements))
        : main[main.length - 1].elements;
      const out = main[main.length - 1].elements;
      const items = [
        ['input', fmt(inp)],
        ['peak', fmt(peak) + '  (' + (peak/inp).toFixed(1) + '×)'],
        ['narrowest inside', fmt(floor) + '  (' + (floor/inp).toFixed(1) + '×)'],
        ['output', fmt(out)]
      ];
      $(dlId).innerHTML = items.map(([k, v]) =>
        '<div><dt>' + k + '</dt><dd class="num">' + v + '</dd></div>').join('');
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
        const all = btn.dataset.stations === 'all';
        fillRow('lrowDif', ld, ldImg, all ? null : keyOnly);
        summary('lsumDif', all ? ld.stations : ld.stations.filter(keyOnly));
        $('lrowDif').classList.toggle('lrow--dense', all);
      });
    });

    /* the verdict, computed from the data rather than asserted */
    const vIn = lv.stations[0].elements, vOut = lv.stations[lv.stations.length-1].elements;
    const dIn = ld.stations[0].elements;
    const dMain = ld.stations.filter(s => (s.branch || 'main') === 'main');
    const dFloor = Math.min(...dMain.slice(1, -1).map(s => s.elements));
    $('verdict').innerHTML =
      'Both networks expand before they contract. Only one ever gets below its input: the ' +
      'encoder ends at <strong>' + vOut + '</strong> numbers, a ' + Math.round(vIn/vOut) +
      '× reduction. The U-Net\'s narrowest interior layer still holds <strong>' + fmt(dFloor) +
      '</strong> — ' + (dFloor/dIn).toFixed(0) + '× the image it was given — and it ends back at ' +
      'full size. It never forms an embedding at all.';
  }

  /* ── I6: meaning has a location ──
     The scatter is coloured by digit class, and clicking anywhere decodes that
     latent vector. The neighbourhood readout is computed from the real encoded
     points, so the claim "this region is mostly 3s" is measured, not asserted. */
  const CLASS_COLOURS = ['#4E79A7','#F28E2B','#E15759','#76B7B2','#59A14F',
                         '#EDC948','#B07AA1','#FF9DA7','#9C755F','#8C8C8C'];

  async function buildRegions(){
    let scatter;
    try { scatter = await loadJSON('vae_scatter.json'); }
    catch (e) { document.querySelector('.regions').style.display = 'none'; return; }

    const pts = scatter.points || [];
    const labelled = !!scatter.labelled && pts.length > 0 && pts[0].length > 2;
    const map = $('regMap'), mctx = map.getContext('2d');
    const W = map.width, H = map.height;

    const toPx = (z1, z2) => [ (z1 - LO) / (HI - LO) * W, (HI - z2) / (HI - LO) * H ];
    const toZ  = (x, y)   => [ LO + x / W * (HI - LO), HI - y / H * (HI - LO) ];

    let sel = [0, 0];

    function paintScatter(){
      mctx.clearRect(0, 0, W, H);
      mctx.fillStyle = '#fff'; mctx.fillRect(0, 0, W, H);

      mctx.strokeStyle = '#EDF1F5'; mctx.lineWidth = 1;      // grid at unit steps
      for (let v = Math.ceil(LO); v <= HI; v++){
        const [gx] = toPx(v, 0), [, gy] = toPx(0, v);
        mctx.beginPath(); mctx.moveTo(gx, 0); mctx.lineTo(gx, H); mctx.stroke();
        mctx.beginPath(); mctx.moveTo(0, gy); mctx.lineTo(W, gy); mctx.stroke();
      }

      for (const p of pts){
        const [x, y] = toPx(p[0], p[1]);
        mctx.fillStyle = labelled ? CLASS_COLOURS[p[2] % 10] : '#8FA0AF';
        mctx.globalAlpha = .5;
        mctx.fillRect(x - 1.4, y - 1.4, 2.8, 2.8);
      }
      mctx.globalAlpha = 1;

      const [sx, sy] = toPx(sel[0], sel[1]);              // selection marker
      mctx.strokeStyle = 'rgba(16,23,32,.35)'; mctx.lineWidth = 1;
      mctx.beginPath(); mctx.moveTo(sx, 0); mctx.lineTo(sx, H);
      mctx.moveTo(0, sy); mctx.lineTo(W, sy); mctx.stroke();
      mctx.beginPath(); mctx.arc(sx, sy, 8, 0, Math.PI * 2);
      mctx.fillStyle = 'rgba(255,255,255,.85)'; mctx.fill();
      mctx.strokeStyle = '#101720'; mctx.lineWidth = 2; mctx.stroke();
    }

    /* what actually lives near the chosen point — measured, not claimed */
    function neighbourhood(z1, z2, k = 40){
      if (!labelled) return null;
      const d = pts.map(p => [(p[0]-z1)**2 + (p[1]-z2)**2, p[2]])
                   .sort((a, b) => a[0] - b[0]).slice(0, k);
      const counts = {};
      for (const [, c] of d) counts[c] = (counts[c] || 0) + 1;
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
      return { digit: +top[0], share: top[1] / d.length,
               spread: Object.keys(counts).length };
    }

    function select(z1, z2){
      z1 = Math.max(LO, Math.min(HI, z1));
      z2 = Math.max(LO, Math.min(HI, z2));
      sel = [z1, z2];

      const col = Math.min(G-1, Math.max(0, Math.round((z1 - LO)/(HI - LO)*(G-1))));
      const row = Math.min(G-1, Math.max(0, Math.round((HI - z2)/(HI - LO)*(G-1))));
      drawTile($('regImage').getContext('2d'), manifoldImg, col, row, mTile, 0);

      $('regZ').textContent = 'z = (' + z1.toFixed(2) + ', ' + z2.toFixed(2) + ')';

      const n = neighbourhood(z1, z2);
      $('regVerdict').innerHTML = n
        ? 'Of the 40 encoded images nearest this point, <strong>' +
          Math.round(n.share * 100) + '% are ' + n.digit + 's</strong>' +
          (n.spread > 1 ? ' — ' + n.spread + ' different digits appear here.' :
                          ' — only one digit appears here.')
        : 'Re-export <code>vae_scatter.json</code> with labels to see which digits live here.';
      paintScatter();
    }

    map.addEventListener('click', e => {
      const r = map.getBoundingClientRect();
      const [z1, z2] = toZ((e.clientX - r.left) / r.width * W,
                           (e.clientY - r.top) / r.height * H);
      select(z1, z2);
    });
    map.addEventListener('keydown', e => {                 // keyboard equivalent
      const step = e.shiftKey ? 0.5 : 0.15;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0],
                      ArrowUp: [0, step], ArrowDown: [0, -step] };
      if (!moves[e.key]) return;
      e.preventDefault();
      select(sel[0] + moves[e.key][0], sel[1] + moves[e.key][1]);
    });

    if (labelled){
      $('regLegend').innerHTML = CLASS_COLOURS.map((c, i) =>
        '<span class="legend__item"><i style="background:' + c + '"></i>' + i + '</span>'
      ).join('');
    }

    select(0.6, 0.6);
  }

  /* ── one update, both channels ── */
  function render(){
    const i    = +scrub.value;                    // 0 … T-1, left→right = noise→image
    const frac = i / (T - 1);

    /* channel A — a new point on a plane */
    const [z1, z2] = latentAt(frac);
    const col = Math.min(G - 1, Math.max(0, Math.round((z1 - LO) / (HI - LO) * (G - 1))));
    const row = Math.min(G - 1, Math.max(0, Math.round((HI - z2) / (HI - LO) * (G - 1))));
    drawTile(vaeImg, manifoldImg, col, row, mTile, 0);
    paintMap(z1, z2);
    $('vaeZ').textContent =
      'z = (' + z1.toFixed(2) + ', ' + z2.toFixed(2) + ')';

    /* channel B — one step along the chain. column i holds t = T-1-i */
    const t = T - 1 - i;
    drawTile(difImg, sheets[view], i, 0, dTile, 0);
    paintSched(t);
    $('difT').textContent = 't = ' + t;

    $('posLabel').textContent = 'step ' + (i + 1) + ' of ' + T;
    renderCompare();
  }

  /* ── I4: the view toggle ──
     Select on [data-view], not .view. The .view class is shared for styling with
     the stations toggle in the layers section; selecting on it reached buttons
     that carry no data-view, set `view` to undefined and blanked Channel B. The
     attribute is the contract, the class is only appearance. */
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('is-active'));
      btn.classList.add('is-active');
      view = btn.dataset.view;
      $('difNote').innerHTML = NOTES[view];
      render();
    });
  });

  scrub.addEventListener('input', render);
  window.addEventListener('resize', paintTicks);

  paintTicks();
  render();
  buildRegions();
  buildLayers();
  veil.classList.add('is-gone');
}

main();