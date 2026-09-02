# Two ways to make an image

An interactive explainer for the relationship between **variational autoencoders** and
**denoising diffusion models**.

Both model families are usually taught separately, and learners come away with two disconnected
mental models. This tool puts them side by side under a single control, so the differences are
visible rather than asserted: where the encoder is learned and where it is fixed, what the network
actually predicts, and how one shared objective underlies both.

> MSc project for *Generative AI for Human-Computer Interaction*, University of Regensburg.
> Supervised by Prof. Bernd Ludwig.

**[→ Open the tool](https://arikaki.github.io/GenAI/)**

---

## What it shows

**One slider drives both panels.** On the VAE side it moves to a new point on a two-dimensional
latent plane. On the diffusion side it advances one step along a 200-step chain in which every
state is the same size as the finished image. The same user action produces two different kinds of
motion, which is the point.

**The diffusion panel can be switched between three views:** the noisy state $x_t$, the network's
actual output $\hat{\epsilon}$, and the clean-image estimate $\hat{x}_0$ recovered from it. Moving
the slider with $\hat{x}_0$ selected shows a blurred average sharpening into a digit, without the
network ever having drawn an image.

**A second section shows where the randomness enters.** One input image encoded five times gives
five slightly different reconstructions, because the encoder emits a distribution rather than a
point. Three reverse runs that differ only in their starting noise resolve into three different
digits.

**A third section compares latent space sizes.** Three VAEs, identical except for how many numbers
the encoder keeps (2, 8, 32), each project a shared set of digits into two dimensions with PCA.
Picking one moves its marker in all three panels at once, showing that the same image lands
differently depending on how much room the encoder was given. A drawing pad lets you sketch your
own digit — encoded live in the browser, run through the same bounding-box-crop and
centre-of-mass-centring the training images went through, checked against a known digit's
precomputed position before the pad is enabled.

## Running it locally

The site is static and needs no build step, but it does need to be served over HTTP: opening
`index.html` directly from the file system will fail, because the page fetches its data as JSON.

```bash
cd docs
python -m http.server 8000
# then open http://localhost:8000
```

## How it is built

Almost everything is precomputed. There is no server and no external API — the one exception is
the free-hand drawing pad, which runs three small encoders live in the browser via a locally
bundled TensorFlow.js, so it still works with no network connection.

The VAE and diffusion models are small, trained from scratch on MNIST. Every other output the page
displays — latent manifolds, reconstructions, forward and reverse trajectories, noise predictions,
the variance schedule, the per-dimension embedding scatters — is precomputed and exported once as
sprite sheets and JSON. The page reads those files and draws to canvas.

Sprite layout is never assumed by the front end: tile size, grid dimensions and column counts are
all read from the accompanying JSON, so re-exporting with different settings does not break the
page.

```text
docs/              served by GitHub Pages
├── index.html
├── app.js
├── style.css
├── vendor/        bundled TensorFlow.js (no CDN, works offline)
├── tfjs/          the three drawing-pad encoders/decoders, converted for the browser
└── data/          precomputed exports (sprite sheets + JSON)
```

`T2_export_vae_cell.py` and `T2_export_diffusion_cell.py` are the original export cells, appended
to the trained model notebooks and run once; they reuse the trained weights and the notebooks' own
schedule functions rather than reimplementing anything. Later ones (`T17`, `T19`, `T20`, `T23`)
follow the same pattern for later sections. `T24_export embeddings.py` is the odd one out — it
trains its own three models from scratch rather than reusing a notebook's, since the comparison
needs encoders none of the other sections have.

## Status

Under active development for a course deadline in September 2026. The interface, the didactic
concept behind it, and its evaluation are all still changing.

## References

- Kingma and Welling (2019). *An Introduction to Variational Autoencoders.* arXiv:1906.02691
- Ho, Jain and Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239
- Lee et al. (2024). *Diffusion Explainer.* IEEE VIS. arXiv:2305.03509
- Bertucci and Endert (2024). *VAE Explainer.* arXiv:2409.09011
- Wang et al. (2021). *CNN Explainer.* IEEE TVCG 27(2), 1396–1406
