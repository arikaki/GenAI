# Two ways to make an image

An interactive explainer for the relationship between **variational autoencoders** and
**denoising diffusion models**.

Both model families are usually taught separately, and learners come away with two disconnected
mental models. This tool puts them side by side under a single control, so the differences are
visible rather than asserted: where the encoder is learned and where it is fixed, what the network
actually predicts, and how one shared objective underlies both.

> MSc project for *Generative AI for Human-Computer Interaction*, University of Regensburg.
> Supervised by Prof. Dr.-Ing. Bernd Ludwig.

**[→ Open the tool](https://arikaki.github.io/GenAI/)**

---

## What it shows

**One slider drives both panels.** On the VAE side it moves to a new point on a two-dimensional
latent plane. On the diffusion side it advances one step along a 200-step chain in which every
state is the same size as the finished image. The same user action produces two different kinds of
motion, which is the point.

**The diffusion panel can be switched between three views** the noisy state $x_t$, the network's
actual output $\hat{\epsilon}$, and the clean-image estimate $\hat{x}_0$ recovered from it. Moving
the slider with $\hat{x}_0$ selected shows a blurred average sharpening into a digit, without the
network ever having drawn an image.

**A second section shows where the randomness enters.** One input image encoded five times gives
five slightly different reconstructions, because the encoder emits a distribution rather than a
point. Three reverse runs that differ only in their starting noise resolve into three different
digits.

## Running it locally

The site is static and needs no build step, but it does need to be served over HTTP opening
`index.html` directly from the file system will fail, because the page fetches its data as JSON.

```bash
cd docs
python -m http.server 8000
# then open http://localhost:8000
```

## How it is built

Nothing is computed at view time. There is no GPU dependency, no server, and no external API.

The VAE and diffusion models are small, trained from scratch on MNIST. Every output the page
displays latent manifolds, reconstructions, forward and reverse trajectories, noise predictions,
the variance schedule is precomputed and exported once as sprite sheets and JSON. The page reads
those files and draws to canvas.

Sprite layout is never assumed by the front end: tile size, grid dimensions and column counts are
all read from the accompanying JSON, so re-exporting with different settings does not break the
page.

```text
docs/              served by GitHub Pages
├── index.html
├── app.js
├── style.css
└── data/          precomputed exports (sprite sheets + JSON)
```

`T2_export_vae_cell.py` and `T2_export_diffusion_cell.py` are the export cells. They are appended
to the trained model notebooks and run once; they reuse the trained weights and the notebooks' own
schedule functions rather than reimplementing anything.

## Status

Under active development for a course deadline in September 2026. The interface, the didactic
concept behind it, and its evaluation are all still changing.

## References

- Kingma and Welling (2019). *An Introduction to Variational Autoencoders.* arXiv:1906.02691
- Ho, Jain and Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239
- Lee et al. (2024). *Diffusion Explainer.* IEEE VIS. arXiv:2305.03509
- Bertucci and Endert (2024). *VAE Explainer.* arXiv:2409.09011
- Wang et al. (2021). *CNN Explainer.* IEEE TVCG 27(2), 1396–1406
