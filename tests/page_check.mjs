// Loads the page in headless Chromium and fails on anything a visitor would
// notice broken: errors, a bundled model the page refused, a failed drawing-pad
// check, numbers that disagree with the exported data, or a layout wider than
// a phone.
//
//   python -m http.server 8000 -d docs     (in another terminal)
//   node tests/page_check.mjs [http://localhost:8000/]
import { chromium } from 'playwright';

const url = process.argv[2] || 'http://localhost:8000/';
const failures = [];
const check = (ok, msg) => { console.log((ok ? 'ok    ' : 'FAIL  ') + msg); if (!ok) failures.push(msg); };
const exists = async path => (await fetch(new URL(path, url), { method: 'HEAD' })).ok;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
// The page probes for models and exports that may not be bundled yet; a 404
// on exactly those is expected. Any other failed request is an error.
const OPTIONAL = ['tfjs/unet/model.json', 'tfjs/vae_decoder/model.json', 'data/reverse_start.png',
                  'data/unet_check.json'];
const errors = [], missing = [];
page.on('response', r => { if (r.status() >= 400) missing.push(new URL(r.url()).pathname); });
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error' && !/^Failed to load resource/.test(m.text())) errors.push(m.text()); });
page.on('console', m => { if (m.type() === 'warning' && /Decoder stays|Live sampling/.test(m.text())) errors.push(m.text()); });

await page.goto(url);
await page.waitForSelector('#loading.is-gone', { state: 'attached', timeout: 60000 });
await page.waitForFunction(() => 'pad' in window.selfChecks, null, { timeout: 120000 }).catch(() => {});

// the live decoder, when bundled, must pass its own check
if (await exists('tfjs/vae_decoder/model.json')) {
  await page.waitForFunction(() => /decoded live/.test(document.getElementById('paSource').textContent),
                             null, { timeout: 60000 }).catch(() => {});
  const src = await page.textContent('#paSource');
  check(/decoded live/.test(src), 'bundled decoder is used live (' + src + ')');
}

const sc = await page.evaluate(() => window.selfChecks);
check(/check OK/.test(await page.textContent('#drawStatus')), 'drawing-pad self-check passes (' +
      JSON.stringify(sc) + ')');

// the equation strip shows the schedule's own coefficients
const sched = await (await fetch(new URL('data/schedule.json', url))).json();
const t0 = sched.T - 1;
check(await page.textContent('#eqA') === sched.sqrt_one_minus_abar[t0].toFixed(2) &&
      await page.textContent('#eqB') === sched.sqrt_alpha_bar[t0].toFixed(2),
      'equation coefficients match schedule.json at t = ' + t0);
check((await page.textContent('#verdictBound')).length > 50, 'bound section is filled in');
check((await page.textContent('#verdictElbo')).length > 50, 'ELBO section is filled in');

// with a bundled U-Net, a click must start a live run (past its self-check)
if (await exists('tfjs/unet/model.json')) {
  await page.click('#newDrawDif');
  const ok = await page.waitForFunction(
    () => /running live: step [2-9]/.test(document.getElementById('pbFrom').textContent),
    null, { timeout: 300000 }).then(() => true, () => false);
  check(ok, 'bundled U-Net passes its self-check and samples live (' +
        JSON.stringify(await page.evaluate(() => window.selfChecks)) + ')');
}

check(errors.length === 0, 'no errors in the console' + (errors.length ? ': ' + errors.join(' | ') : ''));
const unexpected = missing.filter(p => !OPTIONAL.some(o => p.endsWith('/' + o)));
check(unexpected.length === 0, 'every required file loads' +
      (unexpected.length ? ': missing ' + unexpected.join(', ') : ''));

const phone = await browser.newPage({ viewport: { width: 375, height: 800 } });
await phone.goto(url);
await phone.waitForSelector('#loading.is-gone', { state: 'attached', timeout: 60000 });
const [sw, iw] = await phone.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
check(sw <= iw, `no horizontal scroll at 375px (${sw} <= ${iw})`);

await browser.close();
if (failures.length) { console.log(`\n${failures.length} check(s) failed`); process.exit(1); }
console.log('\nall page checks passed');
