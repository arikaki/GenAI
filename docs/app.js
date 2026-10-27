const DATA = 'data/';

/* Self-check tolerances, measured by tests/calibrate_selfchecks.mjs (README,
   "Self-checks"). Each sits between what a correct pipeline scores and what
   realistic mistakes score:
   PAD_TOL   worst drift over one pick per digit, as a fraction of each panel's
             spread. Correct: at most 0.0008. Mildest mistake (no crop): 0.48.
   DEC_TOL   mean per-pixel error of the live decoder against the precomputed
             grid. Bundled or float16 weights: 0.0008; weights perturbed by 10%
             of their spread: 0.012; any other trained decoder: 0.07 to 0.12.
   UNET_TOL  relative RMS error of the live U-Net's noise prediction on the
             exported checks. Float16 conversion: at most 0.008; input on the
             wrong scale: 0.5 or more. Re-run the calibration after retraining. */
const PAD_TOL = 0.05;
const DEC_TOL = 0.01;
const UNET_TOL = 0.05;

const SHEETS = {
  reverse: 'reverse.png',   // x_t   — the noisy state
  eps:     'eps_pred.png',  // eps^  — what the network outputs
  x0:      'x0hat.png'      // x0^   — implied clean image
};

const NOTES = {
  reverse: 'The state at step <em>t</em>, the same size as the finished image, never compressed.',
  eps:     'What the network actually outputs: an estimate of the <em>noise</em> present at step ' +
           '<em>t</em>. Not a picture of a digit.',
  x0:      'The network never draws this. It is computed from &epsilon;&#770; by rearranging the ' +
           'forward equation: blurry early, sharp late.'
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

/* What each self-check measured on this load, for anyone debugging the page
   and for the CI page check. */
window.selfChecks = {};

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
      reconMeta, reconImg, scatter, dec, decImg, startImg;
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
    // x_T itself: reverse.png starts one step in, at x_198. Optional (T30).
    startImg = await optional(loadImage('reverse_start.png'));
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

  const gridCell = () => {
    const N = dec.grid, [gLo, gHi] = dec.range;
    const col = Math.min(N-1, Math.max(0, Math.round((z[0] - gLo)/(gHi - gLo)*(N-1))));
    const row = Math.min(N-1, Math.max(0, Math.round((gHi - z[1])/(gHi - gLo)*(N-1))));
    return { index: row * N + col,
             z: [gLo + col/(N-1)*(gHi - gLo), gHi - row/(N-1)*(gHi - gLo)] };
  };

  /* Decoder activations depend on z. The sprite holds them for a 12x12 grid
     only, so without a live model a click snaps to the nearest grid point.
     When the lab decoder is bundled (T31) and reproduces that grid, every
     station is computed from the exact z instead. */
  let liveDec = null;

  /* Same rendering as T23's to_tile: channel mean for spatial layers, dense
     vectors padded into a square, each tile normalised to its own range. */
  function drawActivation(cv, vals, shape){
    let h, w, img;
    if (shape.length === 3){
      const C = shape[2];
      h = shape[0]; w = shape[1]; img = new Float32Array(h * w);
      for (let p = 0; p < h * w; p++){
        let s = 0;
        for (let c = 0; c < C; c++) s += vals[p * C + c];
        img[p] = s / C;
      }
    } else {
      h = w = Math.ceil(Math.sqrt(vals.length));
      img = new Float32Array(h * w); img.set(vals);
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of img){ if (v < lo) lo = v; if (v > hi) hi = v; }
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    const octx = off.getContext('2d'), id = octx.createImageData(w, h);
    img.forEach((v, p) => {
      const g = hi > lo ? Math.round((v - lo) / (hi - lo) * 255) : 0;
      id.data[p*4] = id.data[p*4+1] = id.data[p*4+2] = g; id.data[p*4+3] = 255;
    });
    octx.putImageData(id, 0, 0);
    const ctx = cv.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(off, 0, 0, cv.width, cv.height);
  }

  function liveActivations(zv){
    let out;
    tf.tidy(() => {
      const ys = liveDec.probe.predict(tf.tensor2d([zv]));
      out = [Float32Array.from(zv)]
        .concat((Array.isArray(ys) ? ys : [ys]).map(t => Float32Array.from(t.dataSync())));
    });
    return out;
  }

  /* Verify before trusting, as the drawing pad does: decode all grid points
     live and compare the final image with the precomputed sprite. A decoder
     from a different training run fails this and the grid stays in use. */
  async function initLiveDecoder(){
    if (!decStations || !decImg || typeof tf === 'undefined') return;
    let model;
    try { model = await tf.loadLayersModel('tfjs/vae_decoder/model.json'); }
    catch (e) { return; }                       // not exported yet: grid it is
    try {
      // by order, not name: Keras renumbers layer names on every rebuild
      const layers = model.layers.filter(l => l.getClassName() !== 'InputLayer');
      if (layers.length !== decStations.length - 1)
        throw new Error('layer count differs from layers_vae_decoder.json');
      const outs = layers.map(l => l.output);
      const probe = tf.model({ inputs: model.inputs, outputs: outs });
      const shapesOk = outs.every((o, k) =>
        o.shape.slice(1).join() === decStations[k + 1].shape.join());
      if (!shapesOk) throw new Error('layer shapes differ from layers_vae_decoder.json');

      const err = await decoderGridError(model);
      window.selfChecks.decoder = err;
      if (err > DEC_TOL) throw new Error('live decoder does not reproduce the grid ' +
                                         '(mean error ' + err.toFixed(3) + ')');
      liveDec = { probe, err };
      paintChannelA();
    } catch (e) {
      console.warn('Decoder stays on the precomputed grid:', e.message);
    }
  }

  /* Mean per-pixel error, on a 0..1 scale, between `model`'s decoding of every
     grid point and the precomputed sprite's last column. */
  async function decoderGridError(model){
    const N = dec.grid, [gLo, gHi] = dec.range, TL = dec.tile;
    const zs = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++)
      zs.push([gLo + c/(N-1)*(gHi - gLo), gHi - r/(N-1)*(gHi - gLo)]);
    const last = tf.tidy(() => model.predict(tf.tensor2d(zs)));
    const imgs = await last.data(); last.dispose();

    const sc = document.createElement('canvas');
    sc.width = decImg.width; sc.height = decImg.height;
    const sctx = sc.getContext('2d');
    sctx.drawImage(decImg, 0, 0);
    const col = decStations.length - 1;
    const [oh, ow] = decStations[col].shape, n = oh * ow;
    let err = 0;
    for (let p = 0; p < zs.length; p++){
      const tile = sctx.getImageData(col * TL, p * TL, TL, TL).data;
      const im = imgs.subarray(p * n, (p + 1) * n);
      let lo = Infinity, hi = -Infinity;
      for (const v of im){ if (v < lo) lo = v; if (v > hi) hi = v; }
      let e = 0;
      for (let y = 0; y < TL; y++) for (let x = 0; x < TL; x++){
        // PIL NEAREST, as used by T23 to scale each tile up to TL
        const sy = Math.floor((y + .5) * oh / TL), sx = Math.floor((x + .5) * ow / TL);
        const v = hi > lo ? (im[sy * ow + sx] - lo) / (hi - lo) : 0;
        e += Math.abs(v - tile[(y * TL + x) * 4] / 255);
      }
      err += e / (TL * TL);
    }
    return err / zs.length;
  }
  window.explainerProbe = Object.assign(window.explainerProbe || {}, { decoderGridError });

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
      $('paSource').textContent = '';
      return;
    }
    const st = decStations[i];
    let paint;
    if (liveDec){
      const acts = liveActivations(z);
      paint = (cv, k) => drawActivation(cv, acts[k], decStations[k].shape);
      $('paSource').textContent = 'every layer below is decoded live from exactly this vector';
    } else {
      const g = gridCell(), p = g.index;
      paint = (cv, k) => drawTile(cv.getContext('2d'), decImg, k, p, dec.tile, 0);
      $('paSource').textContent = 'layers precomputed: shown for the nearest of ' +
        dec.grid * dec.grid + ' grid points, z = (' + g.z[0].toFixed(2) + ', ' +
        g.z[1].toFixed(2) + ')';
    }
    paint($('paImage'), i);
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
      paint(cv, k);
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

  /* The trajectory on show: a stored run (row `seed` of the exported sheets)
     or one sampled live in this browser (`live`, one row per view). */
  let live = null;
  function trajTile(cv, kind, col){
    const ctx = cv.getContext('2d');
    if (live) drawTile(ctx, live[kind], col, 0, dTile, 0);
    else if (kind === 'start') drawTile(ctx, startImg || sheets.reverse, 0, seed, dTile, 0);
    else drawTile(ctx, sheets[kind], col, seed, dTile, 0);
  }

  function paintChannelB(){
    const i = +pbStep.value, t = T - 1 - i;
    trajTile($('pbImage'), view, i);
    $('pbT').textContent = 't = ' + t;
    paintSched(t);
    paintEquation(i, t);
    renderCompare();
  }

  /* x_t as it went into the network at slider position i. Column k of
     reverse.png holds the state *after* step k, so the input is column i-1;
     the very first input, x_T, is a separate export. */
  function paintInput(cv, i){
    if (i > 0) trajTile(cv, 'reverse', i - 1);
    else trajTile(cv, 'start', 0);
  }

  /* The x_t / eps / x0 toggle shows one view at a time, so a sharpening x0
     can be read as the network drawing. This strip shows all three at once,
     with the schedule's numbers, so x0 is visibly arithmetic on the other two. */
  function paintEquation(i, t){
    if (!$('eqXt')) return;
    paintInput($('eqXt'), i);
    trajTile($('eqEps'), 'eps', i);
    trajTile($('eqX0'), 'x0', i);
    $('eqA').textContent = sched.sqrt_one_minus_abar[t].toFixed(2);
    $('eqB').textContent = sched.sqrt_alpha_bar[t].toFixed(2);
    $('eqT').textContent = 't = ' + t;
  }

  /* ── the starting noise tile: x_T itself ── */
  function paintStartNoise(){
    paintInput($('startNoise'), 0);
    $('pbFrom').textContent = live ? 'a fresh draw, run live in your browser'
                                   : 'stored draw ' + (seed + 1) + ' of ' + SEEDS;   // T29
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
    const zLabel = 'z = (' + z[0].toFixed(2) + ', ' + z[1].toFixed(2) + ')';
    $('startZ').textContent = zLabel;
    $('paFrom').textContent = zLabel;          // T29: same vector, said out loud
    const n = neighbourhood();
    $('startVerdict').innerHTML = n
      ? 'Of the 40 encoded images nearest this point, <strong>' +
        Math.round(n.share * 100) + '% are ' + n.digit + 's</strong>' +
        (n.spread > 1 ? ': ' + n.spread + ' different digits appear here.'
                      : ': only one digit appears here.')
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

  /* With the U-Net bundled (T32), a click draws genuinely new noise and runs
     all T steps here. Without it, or if it fails its self-check, the button
     steps through the stored runs, and the note under it says so. */
  const UNET = 'tfjs/unet/model.json';
  let unet = null, unetState = 'unknown';         // unknown | absent | ok | failed
  if (typeof tf !== 'undefined')
    fetch(UNET, { method: 'HEAD' }).then(r => {
      unetState = r.ok ? 'available' : 'absent';
      if (r.ok) $('difNote').textContent = 'The same kind of draw, one value per pixel. ' +
        'Each click draws new noise and runs all ' + T + ' denoising steps live in your ' +
        'browser; the first click downloads the network.';
    }).catch(() => { unetState = 'absent'; });

  function cycleStored(){
    live = null;
    seed = (seed + 1) % SEEDS;
    paintStartNoise();
    paintChannelB();
  }

  /* Verify before trusting: known inputs from the export, compared with the
     network's outputs there. Float16 weights change the outputs slightly;
     a different or broken model changes them completely. */
  async function loadUnet(){
    const model = await tf.loadGraphModel(UNET);
    const { checks } = await loadJSON('unet_check.json');
    let worst = 0;
    for (const c of checks){
      const out = tf.tidy(() => model.execute({ x: tf.tensor4d(c.x, [1, dTile, dTile, 1]),
                                                t: tf.tensor1d([c.t], 'int32') }));
      const got = await out.data(); out.dispose();
      let d2 = 0, e2 = 0;
      c.eps.forEach((v, k) => { d2 += (got[k] - v) ** 2; e2 += v * v; });
      worst = Math.max(worst, Math.sqrt(d2 / e2));
    }
    window.selfChecks.unet = worst;
    if (!(worst < UNET_TOL)) throw new Error('U-Net self-check failed (relative error ' +
                                              worst.toFixed(4) + ')');
    return model;
  }

  const toU8 = v => Math.floor(Math.min(1, Math.max(0, (v + 1) / 2)) * 255);
  function putTile(cv, col, vals){
    const ctx = cv.getContext('2d'), id = ctx.createImageData(dTile, dTile);
    vals.forEach((v, p) => {
      id.data[p*4] = id.data[p*4+1] = id.data[p*4+2] = toU8(v); id.data[p*4+3] = 255;
    });
    ctx.putImageData(id, col * dTile, 0);
  }

  async function runLive(){
    const btn = $('newDrawDif');
    btn.disabled = true;
    try {
      if (!unet){
        $('pbFrom').textContent = 'loading the diffusion network…';
        unet = await loadUnet();
        unetState = 'ok';
      }
      const sheet = () => Object.assign(document.createElement('canvas'),
                                        { width: T * dTile, height: dTile });
      live = { reverse: sheet(), eps: sheet(), x0: sheet(),
               start: Object.assign(document.createElement('canvas'), { width: dTile, height: dTile }) };
      pbStep.disabled = true;
      let x = tf.randomNormal([1, dTile, dTile, 1]);
      putTile(live.start, 0, await x.data());
      paintStartNoise();
      // the same step as diffusion_schedule.reverse_step, sigma_t^2 = beta_t
      for (let k = 0; k < T; k++){
        const t = T - 1 - k;
        const [eps, x0, next] = tf.tidy(() => {
          const e = unet.execute({ x, t: tf.tensor1d([t], 'int32') });
          const x0 = x.sub(e.mul(sched.sqrt_one_minus_abar[t])).div(sched.sqrt_alpha_bar[t]);
          let m = x.sub(e.mul(sched.beta[t] / sched.sqrt_one_minus_abar[t]))
                   .div(Math.sqrt(sched.alpha[t]));
          if (t > 0) m = m.add(tf.randomNormal(x.shape).mul(Math.sqrt(sched.beta[t])));
          return [e, x0, m];
        });
        putTile(live.eps, k, await eps.data());
        putTile(live.x0, k, await x0.data());
        putTile(live.reverse, k, await next.data());
        x.dispose(); eps.dispose(); x0.dispose();
        x = next;
        pbStep.value = k;
        paintChannelB();
        $('pbFrom').textContent = 'running live: step ' + (k + 1) + ' of ' + T;
        await tf.nextFrame();
      }
      x.dispose();
      paintStartNoise();
    } catch (e) {
      console.warn('Live sampling unavailable:', e.message);
      unetState = 'failed';
      $('difNote').textContent = 'The same kind of draw, one value per pixel. The live network ' +
        'could not be used here, so this button steps through three stored draws.';
      cycleStored();
    } finally {
      pbStep.disabled = false;
      btn.disabled = false;
    }
  }

  $('newDrawDif').addEventListener('click', () => {
    if (unetState === 'available' || unetState === 'ok') runLive();
    else cycleStored();
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
  buildBound();
  buildEmbed().then(buildDraw);
  initLiveDecoder();
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
      'More dimensions means better reconstruction and less compression: ' +
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

  /* ── C3: diffusion is trained on the same kind of bound ──
     Written as a VAE with T latent layers, the diffusion bound has one KL
     term per step, and each reduces to a weighted squared error on the
     predicted noise (Ho et al. 2020, eq. 12):
       lambda_t = beta_t^2 / (2 sigma_t^2 alpha_t (1 - abar_t)),
     with sigma_t^2 = beta_t as in the lab sampler. The model was trained on
     the simplified loss, plain MSE on the noise with every step weighted
     equally. Both curves come from this model's own schedule; t = 0 is the
     reconstruction term, not a KL term, and is left out. */
  function buildBound(){
    const c = $('boundChart');
    if (!c) return;
    const g = c.getContext('2d');
    const W = c.width, H = c.height, padL = 46, padR = 16, padT = 16, padB = 26;
    const ts = [];
    for (let t = 1; t < T; t++) ts.push(t);
    const raw = ts.map(t => sched.beta[t] / (2 * sched.alpha[t] * (1 - sched.alpha_bar[t])));
    const mean = raw.reduce((s, v) => s + v, 0) / raw.length;
    const lam = raw.map(v => v / mean);           // scaled so both curves average 1
    const lgLo = Math.floor(Math.log10(Math.min(...lam))),
          lgHi = Math.ceil(Math.log10(Math.max(...lam)));
    // x runs noise -> image like the slider: t = T-1 on the left, t = 1 on the right
    const px = t => padL + (T - 1 - t) / (T - 2) * (W - padL - padR);
    const py = v => (H - padB) - (Math.log10(v) - lgLo) / (lgHi - lgLo) * (H - padT - padB);
    const BOUND_C = '#B4436C', SIMPLE_C = '#16808E';

    g.clearRect(0, 0, W, H);
    g.font = '11px ui-monospace, Consolas, monospace';
    for (let e = lgLo; e <= lgHi; e++){
      const y = py(10 ** e);
      g.strokeStyle = '#E3E8ED'; g.lineWidth = 1;
      g.beginPath(); g.moveTo(padL, y); g.lineTo(W - padR, y); g.stroke();
      g.fillStyle = '#939EA9'; g.textAlign = 'right';
      g.fillText(String(+(10 ** e).toPrecision(1)), padL - 6, y + 4);
    }
    g.strokeStyle = SIMPLE_C; g.lineWidth = 2;
    g.beginPath(); g.moveTo(px(T - 1), py(1)); g.lineTo(px(1), py(1)); g.stroke();
    g.strokeStyle = BOUND_C; g.lineWidth = 2;
    g.beginPath();
    ts.forEach((t, k) => k ? g.lineTo(px(t), py(lam[k])) : g.moveTo(px(t), py(lam[k])));
    g.stroke();
    g.fillStyle = '#939EA9';
    g.textAlign = 'left';  g.fillText('t = ' + (T - 1) + ' · noise', padL, H - 8);
    g.textAlign = 'right'; g.fillText('image · t = 1', W - padR, H - 8);

    const ratio = Math.round(lam[0] / lam[lam.length - 1]);
    const cross = ts.find((t, k) => lam[k] < 1);
    $('verdictBound').innerHTML =
      'Taken literally, the bound would weight the step nearest the image <strong>' + ratio +
      '×</strong> more heavily than the step nearest pure noise. The model on this page was ' +
      'trained the way Ho et al. (2020) recommend: every step weighted equally, which in effect ' +
      'shifts emphasis away from the ' + (cross - 1) + ' steps closest to the image. ' +
      'It is a <strong>reweighted</strong> version of the bound, not the bound itself, which is ' +
      'why this page calls both objectives ELBO-style rather than identical.';
  }

  /* ── T25 Stage 1: the same image, three latent spaces ────
     Three independently trained encoders (dim 2/8/32). Their raw z can't all be
     plotted directly, so each panel shows a PCA projection to 2-D, precomputed
     server-side against the *training* embeddings so picks land in the same
     frame as the background scatter. Selection lives in the digit strip, not
     the panels themselves: a click on an 8-D or 32-D scatter has no unique
     inverse, unlike the raw 2-D map in the start section above. Optional —
     hides itself if T24's export has not been run. */
  async function buildEmbed(){
    let pca, scatter, picks, picksImg;
    try {
      [pca, scatter, picks, picksImg] = await Promise.all([
        loadJSON('embed_pca.json'), loadJSON('embed_scatter.json'),
        loadJSON('embed_picks.json'), loadImage('embed_picks.png')
      ]);
    } catch (e) {
      document.querySelector('.embed').style.display = 'none';
      return;
    }

    const pct = v => v.toLocaleString('en-US', { style: 'percent', maximumFractionDigits: 0 });

    const panelsEl = $('embedPanels');
    panelsEl.innerHTML = '';
    const panels = pca.dims.map(d => {
      const el = document.createElement('div');
      el.className = 'embed__panel';
      const head = document.createElement('div');
      head.className = 'embed__head';
      head.innerHTML =
        '<span class="embed__dim">' + d.dim + (d.dim === 1 ? ' number' : ' numbers') + '</span>' +
        '<span class="embed__stat">' + d.compression + '× compression<br>' +
        pct(d.explained[0] + d.explained[1]) + ' of variance shown</span>';
      const canvas = document.createElement('canvas');
      canvas.width = 300; canvas.height = 300;
      canvas.setAttribute('aria-label',
        'Latent space, ' + d.dim + ' dimensions, projected to 2-D with PCA');

      /* reconstruction thumbnail: what this dim's decoder makes of the live
         drawing's z. Hidden until Stage 2 has something to show — picks have
         no stored z, only their 2-D projection, so this stays empty for them. */
      const reconWrap = document.createElement('div');
      reconWrap.className = 'embed__recon_wrap';
      reconWrap.hidden = true;
      const reconCanvas = document.createElement('canvas');
      reconCanvas.width = reconCanvas.height = 28;
      reconCanvas.className = 'embed__recon';
      const reconLabel = document.createElement('span');
      reconLabel.className = 'embed__recon_label';
      reconLabel.textContent = 'reconstructed';
      reconWrap.append(reconCanvas, reconLabel);

      el.append(head, canvas, reconWrap);
      panelsEl.appendChild(el);
      return {
        dim: d.dim, ctx: canvas.getContext('2d'), w: canvas.width, h: canvas.height,
        reconCtx: reconCanvas.getContext('2d'), reconWrap,
      };
    });

    /* Each dim gets its own padded axis range — the scales are not comparable
       across panels (that is the point: a 32-D space projected to 2-D spreads
       differently than an 8-D one), so a shared range would misrepresent both. */
    const ranges = {};
    pca.dims.forEach(d => {
      const pts = scatter[d.dim];
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      const lo = Math.min(...xs, ...ys), hi = Math.max(...xs, ...ys);
      const pad = (hi - lo) * .08 || 1;
      ranges[d.dim] = [lo - pad, hi + pad];
    });

    let selected = picks.picks[0];

    function paintPanel(p){
      const pts = scatter[p.dim], [lo, hi] = ranges[p.dim];
      const toPx = (a, b) => [(a - lo) / (hi - lo) * p.w, p.h - (b - lo) / (hi - lo) * p.h];
      const ctx = p.ctx;
      ctx.clearRect(0, 0, p.w, p.h);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, p.w, p.h);
      for (const pt of pts){
        const [x, y] = toPx(pt[0], pt[1]);
        ctx.fillStyle = CLASS_COLOURS[pt[2] % 10];
        ctx.globalAlpha = .5;
        ctx.fillRect(x - 1.4, y - 1.4, 2.8, 2.8);
      }
      ctx.globalAlpha = 1;
      const proj = selected.proj[p.dim];
      const [x, y] = toPx(proj[0], proj[1]);
      const isLive = selected.label === null;
      ctx.strokeStyle = 'rgba(16,23,32,.35)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, p.h);
      ctx.moveTo(0, y); ctx.lineTo(p.w, y); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2);
      /* a live drawing has no known label, so it gets a neutral accent marker
         instead of a class colour — visually distinct from the picks, which
         are always a known, coloured digit. */
      ctx.fillStyle = isLive ? '#B4436C' : CLASS_COLOURS[selected.label % 10];
      ctx.fill();
      ctx.strokeStyle = isLive ? '#fff' : '#101720'; ctx.lineWidth = 2; ctx.stroke();
    }
    const paintAll = () => panels.forEach(paintPanel);

    function setRecon(dim, pixels784){
      const p = panels.find(pp => pp.dim === dim);
      if (!p) return;
      const img = p.reconCtx.createImageData(28, 28);
      for (let i = 0; i < 784; i++){
        const v = Math.round(Math.min(1, Math.max(0, pixels784[i])) * 255);
        img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
      }
      p.reconCtx.putImageData(img, 0, 0);
      p.reconWrap.hidden = false;
    }
    const clearRecon = () => panels.forEach(p => { p.reconWrap.hidden = true; });

    const stripEl = $('embedStrip');
    stripEl.innerHTML = '';
    picks.picks.forEach((pk, i) => {
      const btn = document.createElement('button');
      btn.className = 'embed__pick';
      btn.setAttribute('aria-label', 'digit ' + pk.label);
      const cv = document.createElement('canvas');
      cv.width = cv.height = 36;
      drawTile(cv.getContext('2d'), picksImg, i, 0, 28, 1);
      btn.appendChild(cv);
      btn.addEventListener('click', () => {
        selected = pk;
        stripEl.querySelectorAll('.embed__pick').forEach(b => b.classList.remove('is-active'));
        btn.classList.add('is-active');
        clearRecon();
        paintAll();
      });
      stripEl.appendChild(btn);
    });
    stripEl.firstChild.classList.add('is-active');

    const lo = pca.dims[0], hi = pca.dims[pca.dims.length - 1];
    $('embedVerdict').innerHTML =
      'The same image, encoded three times. Its position moves because each encoder learned a ' +
      '<strong>different space</strong>, not because the image changed. Variance explained by ' +
      'the two plotted components falls from <strong>' + pct(lo.explained[0] + lo.explained[1]) +
      '</strong> at ' + lo.dim + ' dimensions to <strong>' + pct(hi.explained[0] + hi.explained[1]) +
      '</strong> at ' + hi.dim + ', and the ' + hi.dim + '-D scatter looks more smeared for that ' +
      'reason alone, not because the model is worse.';

    paintAll();

    return {
      pca, picks, picksImg, scatter,
      select(pk){ selected = pk; paintAll(); },
      setRecon, clearRecon,
    };
  }

  /* ── T25 Stage 2: draw free-hand, encode live ─────────────
     MNIST digits are not raw drawings: each is cropped to its bounding box,
     scaled so the longest side is 20px, then centred in a 28x28 field by
     centre of mass (not bounding-box centre). A canvas drawing fed to the
     encoder without this step lands outside the training distribution — the
     embedding is meaningless. This replicates it in JS. */
  function preprocessCanvas(srcCanvas){
    const sw = srcCanvas.width, sh = srcCanvas.height;
    const src = srcCanvas.getContext('2d').getImageData(0, 0, sw, sh).data;

    let minX = sw, minY = sh, maxX = -1, maxY = -1;
    for (let y = 0; y < sh; y++){
      for (let x = 0; x < sw; x++){
        if (src[(y * sw + x) * 4] > 12){   // red channel: white strokes on black
          if (x < minX) minX = x; if (x > maxX) maxX = x;
          if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return null;   // nothing drawn

    const bw = maxX - minX + 1, bh = maxY - minY + 1;
    const scale = 20 / Math.max(bw, bh);
    const rw = Math.max(1, Math.round(bw * scale)), rh = Math.max(1, Math.round(bh * scale));

    const off = document.createElement('canvas');
    off.width = rw; off.height = rh;
    const octx = off.getContext('2d');
    octx.imageSmoothingEnabled = true;
    octx.drawImage(srcCanvas, minX, minY, bw, bh, 0, 0, rw, rh);
    const scaled = octx.getImageData(0, 0, rw, rh).data;

    let sumX = 0, sumY = 0, sumM = 0;
    for (let y = 0; y < rh; y++){
      for (let x = 0; x < rw; x++){
        const v = scaled[(y * rw + x) * 4] / 255;
        sumX += x * v; sumY += y * v; sumM += v;
      }
    }
    const cx = sumM > 0 ? sumX / sumM : rw / 2;
    const cy = sumM > 0 ? sumY / sumM : rh / 2;
    const offX = Math.round(14 - cx), offY = Math.round(14 - cy);

    const out = new Float32Array(28 * 28);
    for (let y = 0; y < rh; y++){
      const dy = y + offY;
      if (dy < 0 || dy >= 28) continue;
      for (let x = 0; x < rw; x++){
        const dx = x + offX;
        if (dx < 0 || dx >= 28) continue;
        out[dy * 28 + dx] = scaled[(y * rw + x) * 4] / 255;
      }
    }
    return out;
  }

  function project(zRaw, mean, components){
    const centered = zRaw.map((v, i) => v - mean[i]);
    return [0, 1].map(k => components[k].reduce((s, c, i) => s + c * centered[i], 0));
  }

  async function encode(pixels784, model){
    const t = tf.tensor4d(pixels784, [1, 28, 28, 1]);
    const zt = model.predict(t);
    const z = Array.from(await zt.data());
    t.dispose(); zt.dispose();
    return z;
  }

  async function decode(z, model){
    const t = tf.tensor2d([z]);
    const rt = model.predict(t);
    const pixels = await rt.data();
    t.dispose(); rt.dispose();
    return pixels;
  }

  async function buildDraw(embed){
    const drawSection = document.querySelector('.embed__interact');
    if (!embed || typeof tf === 'undefined'){ drawSection.style.display = 'none'; return; }

    let models;
    try {
      models = {};
      for (const d of embed.pca.dims){
        models[d.dim] = await tf.loadLayersModel('tfjs/dim' + d.dim + '/encoder/model.json');
      }
    } catch (e) {
      drawSection.style.display = 'none';
      return;
    }

    /* Reconstructions are the first thing on the cut list — nice, not
       load-bearing — so a failure here just leaves the thumbnails hidden
       rather than taking down live positioning. */
    let decModels = null;
    try {
      decModels = {};
      for (const d of embed.pca.dims){
        decModels[d.dim] = await tf.loadLayersModel('tfjs/dim' + d.dim + '/decoder/model.json');
      }
    } catch (e) { decModels = null; }

    const statusEl = $('drawStatus');
    const byDim = {};
    embed.pca.dims.forEach(d => { byDim[d.dim] = d; });

    async function liveProject(pixels784){
      const proj = {};
      for (const d of embed.pca.dims){
        const z = await encode(pixels784, models[d.dim]);
        proj[d.dim] = project(z, d.mean, d.components);
      }
      return proj;
    }

    async function liveProjectAndReconstruct(pixels784){
      const proj = {};
      for (const d of embed.pca.dims){
        const z = await encode(pixels784, models[d.dim]);
        proj[d.dim] = project(z, d.mean, d.components);
        if (decModels) embed.setRecon(d.dim, await decode(z, decModels[d.dim]));
      }
      return proj;
    }

    /* Drift is measured relative to each panel's spread (RMS distance of its
       scatter from the centre), since the 2-, 8- and 32-D projections have
       different scales. */
    const spread = {};
    for (const d of embed.pca.dims){
      const pts = embed.scatter[d.dim] || [];
      const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length,
            my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
      spread[d.dim] = Math.sqrt(pts.reduce((s, p) => s + (p[0]-mx)**2 + (p[1]-my)**2, 0) / pts.length);
    }
    function relDrift(proj, ref){
      let worst = 0;
      for (const d of embed.pca.dims){
        const [ex, ey] = proj[d.dim], [rx, ry] = ref[d.dim];
        worst = Math.max(worst, Math.hypot(ex - rx, ey - ry) / spread[d.dim]);
      }
      return worst;
    }
    window.explainerProbe = Object.assign(window.explainerProbe || {}, {
      preprocessCanvas, liveProject, relDrift, picks: embed.picks.picks, picksImg: embed.picksImg,
      testPicks: () => testPicks.map(([, i]) => i)
    });

    /* Verify before trusting: run known MNIST picks through the exact same
       canvas -> preprocess -> encode -> project path used for live drawing,
       and compare against their precomputed positions. If they disagree, the
       preprocessing is wrong and every live draw downstream is noise. One
       pick per digit: a single, nearly symmetric digit cannot tell centring
       by centre of mass from centring by bounding box. */
    const testPicks = embed.picks.picks
      .map((pk, i) => [pk, i])
      .filter(([pk], k, all) => all.findIndex(([q]) => q.label === pk.label) === k);
    async function selfTest(){
      let worst = 0;
      for (const [pick, i] of testPicks){
        const off = document.createElement('canvas');
        off.width = off.height = 140;
        const octx = off.getContext('2d');
        octx.fillStyle = '#000'; octx.fillRect(0, 0, 140, 140);
        octx.imageSmoothingEnabled = false;
        // paste the 28px tile at an arbitrary offset and size, as a drawing would
        // land. 80px or more: smaller nearest-neighbour scalings distort the digit.
        octx.drawImage(embed.picksImg, i * 29, 0, 28, 28, 30, 40, 80, 80);
        worst = Math.max(worst, relDrift(await liveProject(preprocessCanvas(off)), pick.proj));
      }
      return worst;
    }

    const drift = await selfTest();
    window.selfChecks.pad = drift;
    const ok = drift < PAD_TOL;
    statusEl.textContent = ok
      ? 'preprocessing check OK (drift ' + (drift * 100).toFixed(0) + '% of the spread, vs a known digit)'
      : 'preprocessing check failed (drift ' + (drift * 100).toFixed(0) + '% of the spread): ' +
        'live positions may be unreliable';
    statusEl.classList.add(ok ? 'is-ok' : 'is-bad');
    if (!ok) return;   // don't wire a pipeline known to be wrong

    const pad = $('drawPad'), pctx = pad.getContext('2d');
    pctx.fillStyle = '#000'; pctx.fillRect(0, 0, pad.width, pad.height);
    let drawing = false, dirty = false, raf = null;

    function posFromEvent(e){
      const r = pad.getBoundingClientRect();
      const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      return [cx / r.width * pad.width, cy / r.height * pad.height];
    }

    function strokeTo(x, y, first){
      pctx.strokeStyle = '#fff'; pctx.lineWidth = 14;
      pctx.lineCap = 'round'; pctx.lineJoin = 'round';
      if (first){ pctx.beginPath(); pctx.moveTo(x, y); }
      pctx.lineTo(x, y); pctx.stroke();
      dirty = true;
    }

    async function onDirty(){
      if (!dirty) return;
      dirty = false;
      const pixels = preprocessCanvas(pad);
      if (!pixels) return;
      const proj = await liveProjectAndReconstruct(pixels);
      embed.select({ label: null, proj });
    }

    function loop(){ onDirty(); raf = requestAnimationFrame(loop); }

    pad.addEventListener('pointerdown', e => {
      drawing = true; pad.setPointerCapture(e.pointerId);
      const [x, y] = posFromEvent(e); strokeTo(x, y, true);
      if (!raf) loop();
    });
    pad.addEventListener('pointermove', e => {
      if (!drawing) return;
      const [x, y] = posFromEvent(e); strokeTo(x, y, false);
    });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach(ev =>
      pad.addEventListener(ev, () => { drawing = false; }));

    $('drawClear').addEventListener('click', () => {
      pctx.fillStyle = '#000'; pctx.fillRect(0, 0, pad.width, pad.height);
      embed.clearRecon();
      embed.select(embed.picks.picks[0]);
    });
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
      '</strong>, ' + (dFloor/dIn).toFixed(0) + '× the image it was given, and it ends back at ' +
      'full size. It never forms an embedding at all.';
  }
}

main();
