# Two ways to make an image

[![checks](https://github.com/arikaki/GenAI/actions/workflows/ci.yml/badge.svg)](https://github.com/arikaki/GenAI/actions/workflows/ci.yml)

An interactive explainer for the relationship between **variational autoencoders** and
**denoising diffusion models**.

The two families are usually taught in separate blocks, and learners come away with two
disconnected mental models, most commonly the belief that diffusion "encodes into a latent space"
the way a VAE does. This tool is built around three specific misconceptions and shows, from the
trained models themselves, where the two families genuinely diverge.

> MSc project for *Generative AI for Human-Computer Interaction*, University of Regensburg.
> Supervised by Prof. Dr. Bernd Ludwig.

**[→ Open the tool](https://arikaki.github.io/GenAI/)**

---

## What it shows

**Draw a random vector.** Both models start by drawing from N(0, I). The page shows that draw
twice: as two numbers for the VAE, and as a 32×32 field for the diffusion model. The draws are
independent and each has its own control. The VAE's draws a fresh Gaussian sample; the
diffusion one draws fresh noise and runs all 200 denoising steps live in the browser, falling back
to three stored runs if the network is not bundled or fails its self-check.
The difference in their *size* is the first thing the tool is trying to teach. The draw is chosen
on a map of 5,000 MNIST images (drawn from the training and test sets together), coloured by digit
class, which reports the composition of whatever region you click.

**Variational Autoencoder.** The chosen vector stepping through the decoder layers, with the real
activation and shape at each stage, until it is an image. The layers are decoded live from the
exact vector; the page says so under the heading, or names the precomputed grid point it falls
back to if the live decoder is missing or fails its self-check.

**Diffusion Model.** Two hundred denoising steps, with a toggle between the noisy state x_t, the
network's actual output ε̂, and the clean-image estimate x̂₀ recovered from it. Dragging through
the steps with x̂₀ selected shows a blurred average sharpening into a digit, an image the network
never draws, only implies. Because a toggle shows one view at a time, a strip beneath it shows
all three at once as the equation that links them, x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) ÷ √ᾱ_t, with the
schedule's numbers at the current step: the network outputs only ε̂, and x̂₀ is arithmetic on it.

**Where the randomness lives.** One image encoded five times gives five slightly different
reconstructions, because the encoder emits a distribution rather than a point. Three reverse runs
differing only in their starting noise resolve into three different digits.

**How the input is compressed.** Every layer of both networks, probed with a real forward pass.
Tile size follows the spatial dimension; bar height follows the number of values held, on a log
scale shared by both models. The findings are not visible in a block diagram:

| | input | peak | narrowest interior | output |
| --- | --- | --- | --- | --- |
| VAE encoder | 784 | 6,272 (8×) | **2** | 2 (an embedding) |
| Diffusion U-Net | 1,024 | 65,536 (64×) | **4,096 (4×)** | 1,024 (image-shaped) |

Both networks expand before they contract. Only the VAE ever gets below its input. The U-Net's
narrowest interior layer still holds four times the image it was given, and it ends back at full
size; it never forms an embedding at all.

**What is being optimised.** The reconstruction and KL terms over training epochs, pulling against
each other. The KL term rising while the total falls is not a failure; it is the encoder buying
reconstruction accuracy by moving the posterior away from the prior.

**Diffusion is trained on the same kind of bound.** The two bounds term by term (latent, encoder,
reconstruction, prior, per-step terms), treating diffusion as a VAE with 200 latent layers and a
fixed encoder. Beneath it, computed from the model's own schedule, the weight the bound would give
each timestep against the equal weighting of the simplified loss the model was actually trained
with, which is why the page calls both objectives ELBO-style rather than identical.

**Changing the size of the latent space.** The same model trained at four latent sizes (1, 2, 8
and 32) reconstructing the same digits. More dimensions means better reconstruction and less
compression; there is no correct answer, only a trade-off. Two dimensions is used everywhere else
on the page for one reason: it is the largest latent space that can be *drawn*.

## Running it locally

The site is static and needs no build step, but it must be served over HTTP: opening
`index.html` from the file system will fail, because the page fetches its data as JSON.

```bash
cd docs
python -m http.server 8000
# then open http://localhost:8000
```

## How it is built

Most of the page is precomputed: the models are trained offline, and their latent manifolds,
reconstructions, trajectories, noise predictions, layer activations and schedule are exported as
sprite sheets and JSON that the page draws to canvas. Three things run live in the browser as
TensorFlow.js models: the encoders behind the drawing pad, the VAE decoder, and the diffusion
U-Net. Sprite layout is never assumed by the front end: tile sizes, grid dimensions, station lists
and column counts are all read from the accompanying JSON.

```
docs/                    served by GitHub Pages
├── index.html, app.js, style.css
├── data/                precomputed exports (sprite sheets + JSON)
└── tfjs/                the live models
diffusion_schedule.py    the diffusion schedule, forward process and sampler (NumPy)
T32_train_diffusion.py   trains the U-Net and runs every diffusion export
T33_train_vae.py         trains the VAE and runs every VAE export
T*_export_*.py           the individual export steps, run by T32/T33
tests/                   maths tests, page check, self-check calibration
```

### Reproducing the models

```bash
python T33_train_vae.py          # VAE, 30 epochs, then every VAE export (minutes on a GPU)
python T32_train_diffusion.py    # U-Net, 10 epochs, then every diffusion export (GPU recommended)
```

Each script trains, saves its weights under `lab/`, reloads them cold to prove they round-trip,
runs every export against those same weights, and prints the `tensorflowjs_converter` command for
its browser model (run it in the venv from `T25_setup_tfjs_venv.sh`; the VAE decoder also needs
`T25_fix_tfjs_topology.py`). One process per model means every figure, the loss curves and the
live model come from the same run. `--smoke` runs the whole pipeline on a sliver of data in a
minute, to check it rather than the model; `T32 --skip-train` re-runs only the exports.

The latent-size comparison (`T20`) and the three drawing-pad encoders (`T24`) are trained by their
own scripts.

### Self-checks

Every live model is checked against the precomputed exports before the page uses it, and the page
falls back to the precomputed version if a check fails. Each tolerance was set by
`tests/calibrate_selfchecks.mjs`, which measures, inside the real page, what each check reports for
a correct pipeline and for realistic mistakes:

| Check | Correct | Mistakes | Tolerance |
| --- | --- | --- | --- |
| Drawing pad: worst drift over one pick per digit, as a fraction of each panel's spread | ≤ 0.0008 | 0.48 (no crop) to 3.9 (inverted) | 0.05 |
| Live decoder: mean per-pixel error against the 12×12 grid | 0.0008 (float32 or float16) | 0.012 (weights perturbed by 10%); 0.07–0.12 (another trained decoder) | 0.01 |
| Live U-Net: relative RMS error of ε̂ on exported inputs | ≤ 0.008 (float16) | ≥ 0.5 (input on the wrong scale) | 0.05 |

The pad check uses one pick of each digit because a single, nearly symmetric digit cannot tell
centring by centre of mass (what MNIST does) from centring by bounding box. Re-run the calibration
after retraining.

### Tests and CI

`pytest tests` checks the diffusion maths without a model: given the true noise, the reverse step
must equal the true posterior q(x_{t−1} | x_t, x₀), and 200 such steps from pure noise must land
exactly on x₀. `tests/page_check.mjs` loads the page in headless Chromium and fails on console
errors, a bundled model the page refused, a failed drawing-pad check, coefficients that disagree
with `schedule.json`, or horizontal scrolling at phone width. GitHub Actions runs both on every
push.

### Projection: PCA, not t-SNE

Embeddings from latent spaces larger than two dimensions are projected to 2-D with PCA.
t-SNE has no out-of-sample extension; a newly encoded digit cannot be placed on an existing t-SNE
map without refitting the whole projection, which moves every other point. PCA projects a new
embedding with a single matrix multiplication, which is what an interactive feature requires.

## Changes since the study

The study and the graded report describe the tool as it was before these changes (commit
`01d004d` on `main`). Since then:

- **Diffusion schedule.** The lab notebook used β from 1e-4 to 0.02 over 200 steps, values meant
  for 1,000 steps. It left ᾱ_T = 0.13, so 36% of the image's amplitude survived to the last step,
  while sampling started from pure noise. The schedule is now rescaled to 200 steps and ends at
  ᾱ_T = 3×10⁻⁵.
- **Sampler indexing.** The notebook shifted ᾱ by one, so its first timestep had no noise at all,
  and its reverse step used the α of the neighbouring step. The second error was numerically tiny
  (under 0.01% per step), but the tests now pin the maths down.
- **Diffusion checkpoint.** The notebook's `tf.train.Checkpoint` did not track layers kept in
  nested lists, so it saved 10.4M of the U-Net's 32.3M parameters and none of its down or up
  blocks. Weights are now saved with `save_weights` and checked by a cold reload.
- **One run per model.** The VAE loss curves came from an earlier run than its figures; both now
  come from the same run, as do the live models. VAE layers are named explicitly, so the names on
  the page no longer depend on the notebook session.
- **Live models.** The decoder steps through the exact z, not the nearest of 144 grid points, and
  "new noise image" samples a genuinely new run instead of cycling three stored ones.
- **Interface.** ε̂ and x̂₀ are shown together as the equation that links them, and a section
  compares the two models' bounds term by term (the report's C3 gap).

## Known approximations

- **MNIST-scale models.** Sample quality is modest and visible. The tool is about the concepts,
  not about the behaviour of large generative models.
- **The live U-Net is heavy.** It has 32M parameters, about 65 MB at half precision, downloaded
  on the first click, and a run is 200 passes through it. Speed depends on the visitor's GPU;
  without one it is slow (about six minutes in software-rendered headless Chromium). The three
  stored runs in "Where the randomness lives" stay precomputed.
- **x̂₀ at the noisiest steps.** With ᾱ_T near zero, x̂₀ divides the network's residual error by
  √ᾱ_t, down to 0.0055, so at the first few steps it amplifies small errors in ε̂ and is clipped to
  the image range.
- **Two PCA components explain less of a 32-dimensional space** than of an 8-dimensional one, so
  the higher-dimensional scatters look more smeared. That is the projection, not the model.

## References

- Kingma and Welling (2019). *An Introduction to Variational Autoencoders.* arXiv:1906.02691
- Ho, Jain and Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239
- Lee et al. (2024). *Diffusion Explainer.* IEEE VIS. arXiv:2305.03509
- Bertucci and Endert (2024). *VAE Explainer.* arXiv:2409.09011
- Wang et al. (2021). *CNN Explainer.* IEEE TVCG 27(2), 1396–1406
