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

  /* ── I4: the view toggle ── */
  document.querySelectorAll('.view').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.view').forEach(b => b.classList.remove('is-active'));
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
  veil.classList.add('is-gone');
}

main();