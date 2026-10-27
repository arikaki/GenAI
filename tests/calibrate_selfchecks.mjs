// Calibrates the page's three self-check tolerances (PAD_TOL, DEC_TOL, UNET_TOL
// in docs/app.js) by measuring, inside the real page, what each check reports
// for a correct pipeline and for realistic ways of getting it wrong. A
// tolerance is sound when it sits clearly between the two.
//
//   python -m http.server 8000 -d docs     (in another terminal)
//   node tests/calibrate_selfchecks.mjs [http://localhost:8000/]
//
// Needs the `playwright` package and its Chromium. Re-run after retraining.
import { chromium } from 'playwright';

const url = process.argv[2] || 'http://localhost:8000/';
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(url);
await page.waitForFunction(() => window.explainerProbe && window.explainerProbe.relDrift &&
                                 'pad' in window.selfChecks, null, { timeout: 120000 });

const stats = xs => {
  const s = [...xs].sort((a, b) => a - b), q = p => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { n: s.length, min: +s[0].toFixed(4), median: +q(0.5).toFixed(4),
           p99: +q(0.99).toFixed(4), max: +s[s.length - 1].toFixed(4) };
};

/* ── drawing pad ─────────────────────────────────────────────
   The statistic is the page's own: the worst drift over its test picks (one
   per digit), for one placement. Correct: the page's preprocessCanvas, over
   many sizes and positions. Broken: the same canvases through preprocessing
   with one realistic mistake each. Sizes start at 80px: nearest-neighbour
   upscaling a 28px tile to 40 or 60px duplicates rows unevenly and changes
   the digit itself, which is not what the check is meant to detect. */
const pad = await page.evaluate(async () => {
  const P = window.explainerProbe;
  let seed = 1;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

  function pasted(pick, size){
    const c = document.createElement('canvas'); c.width = c.height = 140;
    const g = c.getContext('2d');
    g.fillStyle = '#000'; g.fillRect(0, 0, 140, 140); g.imageSmoothingEnabled = false;
    const x = Math.floor(rand() * (140 - size)), y = Math.floor(rand() * (140 - size));
    g.drawImage(P.picksImg, pick * 29, 0, 28, 28, x, y, size, size);
    return c;
  }
  function bbox(c){
    const d = c.getContext('2d').getImageData(0, 0, 140, 140).data;
    let x0 = 140, y0 = 140, x1 = -1, y1 = -1;
    for (let y = 0; y < 140; y++) for (let x = 0; x < 140; x++)
      if (d[(y * 140 + x) * 4] > 12){ x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
  }
  function render(c, [bx, by, bw, bh], w, h, ox, oy){
    const o = document.createElement('canvas'); o.width = o.height = 28;
    const g = o.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, 28, 28);
    g.imageSmoothingEnabled = true;
    g.drawImage(c, bx, by, bw, bh, ox, oy, w, h);
    const d = g.getImageData(0, 0, 28, 28).data;
    return Float32Array.from({ length: 784 }, (_, k) => d[k * 4] / 255);
  }
  const fit20 = b => { const s = 20 / Math.max(b[2], b[3]); return [Math.round(b[2] * s), Math.round(b[3] * s)]; };
  const broken = {
    'whole canvas, no crop': c => render(c, [0, 0, 140, 140], 28, 28, 0, 0),
    'cropped, not centred': c => { const b = bbox(c), [w, h] = fit20(b); return render(c, b, w, h, 0, 0); },
    'stretched to 28x28': c => render(c, bbox(c), 28, 28, 0, 0),
    'centred by bounding box': c => { const b = bbox(c), [w, h] = fit20(b);
                                      return render(c, b, w, h, Math.round(14 - w / 2), Math.round(14 - h / 2)); },
    'colours inverted': c => P.preprocessCanvas(c).map(v => 1 - v),
    'transposed': c => { const p = P.preprocessCanvas(c);
                         return Float32Array.from({ length: 784 }, (_, k) => p[(k % 28) * 28 + Math.floor(k / 28)]); },
  };

  const variants = { correct: c => P.preprocessCanvas(c), ...broken };
  const out = Object.fromEntries(Object.keys(variants).map(k => [k, []]));
  for (const size of [80, 100, 130]){
    for (let r = 0; r < 10; r++){
      const worst = Object.fromEntries(Object.keys(variants).map(k => [k, 0]));
      for (const i of P.testPicks()){
        const c = pasted(i, size);
        for (const [k, f] of Object.entries(variants))
          worst[k] = Math.max(worst[k], P.relDrift(await P.liveProject(f(c)), P.picks[i].proj));
      }
      for (const k of Object.keys(variants)) out[k].push(worst[k]);
    }
  }
  return out;
});
console.log('\nDrawing pad: worst drift over the three panels, as a fraction of each panel\'s spread');
for (const [k, v] of Object.entries(pad)) console.log(' ', k.padEnd(26), stats(v));
console.log('  page self-check reported   ', await page.evaluate(() => window.selfChecks.pad));

/* ── live decoder ────────────────────────────────────────────
   Correct: the bundled decoder, and the same weights rounded to float16 as a
   quantised export would. Wrong: weights perturbed by a fraction of each
   tensor's spread, and a different trained decoder with the same architecture
   (the 2-D comparison model). */
const dec = await page.evaluate(async () => {
  const P = window.explainerProbe;
  if (!P.decoderGridError) return null;
  const base = await tf.loadLayersModel('tfjs/vae_decoder/model.json');
  const orig = base.getWeights().map(w => w.clone());
  const res = { bundled: await P.decoderGridError(base) };
  base.setWeights(orig.map(w => tf.tidy(() => w.cast('float32'))));
  const f16 = orig.map(w => tf.tensor(Float32Array.from(w.dataSync(), v => {
    const b = new Float32Array([v]), i = new Uint32Array(b.buffer);
    i[0] = (i[0] + 0x1000) & 0xffffe000;          // round the mantissa to float16's 10 bits
    return b[0];
  }), w.shape));
  base.setWeights(f16);
  res['float16 weights'] = await P.decoderGridError(base);
  for (const sigma of [0.003, 0.01, 0.03, 0.1, 0.3]){
    base.setWeights(orig.map(w => tf.tidy(() => {
      const sd = tf.moments(w).variance.sqrt();
      return w.add(tf.randomNormal(w.shape, 0, 1, 'float32', 7).mul(sd).mul(sigma));
    })));
    res['perturbed ' + sigma] = await P.decoderGridError(base);
  }
  try {
    res['other trained decoder'] = await P.decoderGridError(
      await tf.loadLayersModel('tfjs/dim2/decoder/model.json'));
  } catch (e) {}
  return res;
});
console.log('\nLive decoder: mean per-pixel error against the precomputed grid (0..1)');
if (dec) for (const [k, v] of Object.entries(dec)) console.log(' ', k.padEnd(26), v.toFixed(4));
else console.log('  (no bundled decoder)');

/* ── live U-Net ──────────────────────────────────────────────
   Correct: the bundled (float16) network on the exported checks. Wrong: the
   same network given the timestep off by a few steps, or the input on the
   wrong scale -- the mistakes a sampler bug would make. */
const unet = await page.evaluate(async () => {
  let model, checks;
  try {
    model = await tf.loadGraphModel('tfjs/unet/model.json');
    ({ checks } = await (await fetch('data/unet_check.json')).json());
  } catch (e) { return null; }
  const rel = (c, x, t) => tf.tidy(() => {
    const got = model.execute({ x: tf.tensor4d(x, [1, 32, 32, 1]), t: tf.tensor1d([t], 'int32') });
    const want = tf.tensor(c.eps, got.shape);
    return got.sub(want).square().sum().div(want.square().sum()).sqrt().dataSync()[0];
  });
  const res = { bundled: [], 't off by 1': [], 't off by 5': [], 'input scaled to [0,1]': [] };
  for (const c of checks){
    res.bundled.push(rel(c, c.x, c.t));
    res['t off by 1'].push(rel(c, c.x, Math.max(0, c.t - 1)));
    res['t off by 5'].push(rel(c, c.x, Math.max(0, c.t - 5)));
    res['input scaled to [0,1]'].push(rel(c, c.x.map(v => (v + 1) / 2), c.t));
  }
  return res;
});
console.log('\nLive U-Net: relative RMS error of the predicted noise on the exported checks');
if (unet) for (const [k, v] of Object.entries(unet)) console.log(' ', k.padEnd(26), v.map(x => x.toFixed(4)).join('  '));
else console.log('  (no bundled U-Net)');

await browser.close();
