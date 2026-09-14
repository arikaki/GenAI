# Two ways to make an image

An interactive explainer for the relationship between **variational autoencoders** and
**denoising diffusion models**.

The two families are usually taught in separate blocks, and learners come away with two
disconnected mental models — most commonly the belief that diffusion "encodes into a latent space"
the way a VAE does. This tool is built around three specific misconceptions and shows, from the
trained models themselves, where the two families genuinely diverge.

> MSc project for *Generative AI for Human-Computer Interaction*, University of Regensburg.
> Supervised by Prof. Dr.-Ing. Bernd Ludwig.

**[→ Open the tool](https://arikaki.github.io/GenAI/)**

---

## What it shows

**Draw a random vector.** Both models start by drawing from N(0, I). The page shows that draw
twice — as two numbers for the VAE, and as a 32×32 field for the diffusion model. The draws are
independent and can be re-rolled separately; the difference in their *size* is the first thing the
tool is trying to teach. The draw is chosen on a map of 5000 encoded test images, coloured by
digit class, which reports the composition of whatever region you click.

**Variational Autoencoder.** The chosen vector stepping through the decoder layers, with the real
activation and shape at each stage, until it is an image.

**Diffusion Model.** Two hundred denoising steps, with a toggle between the noisy state x_t, the
network's actual output ε̂, and the clean-image estimate x̂₀ recovered from it. Dragging through
the steps with x̂₀ selected shows a blurred average sharpening into a digit — an image the network
never draws, only implies.

**Where the randomness lives.** One image encoded five times gives five slightly different
reconstructions, because the encoder emits a distribution rather than a point. Three reverse runs
differing only in their starting noise resolve into three different digits.

**How the input is compressed.** Every layer of both networks, probed with a real forward pass.
Tile size follows the spatial dimension; bar height follows the number of values held, on a log
scale shared by both models. The findings are not visible in a block diagram:

| | input | peak | narrowest interior | output |
| --- | --- | --- | --- | --- |
| VAE encoder | 784 | 6,272 (8×) | **2** | 2 — an embedding |
| Diffusion U-Net | 1,024 | 65,536 (64×) | **4,096 (4×)** | 1,024 — image-shaped |

Both networks expand before they contract. Only the VAE ever gets below its input. The U-Net's
narrowest interior layer still holds four times the image it was given, and it ends back at full
size — it never forms an embedding at all.

**What is being optimised.** The reconstruction and KL terms over training epochs, pulling against
each other. The KL term rising while the total falls is not a failure; it is the encoder buying
reconstruction accuracy by moving the posterior away from the prior.

**Changing the size of the latent space.** The same model trained at four latent sizes — 1, 2, 8
and 32 — reconstructing the same digits. More dimensions means better reconstruction and less
compression; there is no correct answer, only a trade-off. Two dimensions is used everywhere else
on the page for one reason: it is the largest latent space that can be *drawn*.

## Running it locally

The site is static and needs no build step, but it must be served over HTTP — opening
`index.html` from the file system will fail, because the page fetches its data as JSON.

```bash
cd docs
python -m http.server 8000
# then open http://localhost:8000
```

## How it is built

Model outputs are precomputed. The VAE and diffusion models are small, trained from scratch on
MNIST; latent manifolds, reconstructions, forward and reverse trajectories, noise predictions,
layer activations and the variance schedule are all exported once as sprite sheets and JSON. The
page reads those files and draws to canvas.

Sprite layout is never assumed by the front end: tile size, grid dimensions, station lists and
column counts are all read from the accompanying JSON, so re-exporting with different settings
does not break the page.

```
docs/              served by GitHub Pages
├── index.html
├── app.js
├── style.css
└── data/          precomputed exports (sprite sheets + JSON)
```

### Projection: PCA, not t-SNE

Embeddings from latent spaces larger than two dimensions are projected to 2-D with PCA.
t-SNE has no out-of-sample extension — a newly encoded digit cannot be placed on an existing t-SNE
map without refitting the whole projection, which moves every other point. PCA projects a new
embedding with a single matrix multiplication, which is what an interactive feature requires.

### Export cells

Each is appended to the trained model notebook and run once. They reuse the trained weights and
the notebooks' own schedule functions rather than reimplementing anything.

| Cell | Produces |
| --- | --- |
| `T2_export_vae_cell.py` | latent manifold, scatter, reconstructions, interpolations, loss history |
| `T2_export_diffusion_cell.py` | schedule, forward process, reverse trajectories, ε̂ and x̂₀ views |
| `T17_export_vae_layers.py` | encoder layer activations and shapes |
| `T17_export_diffusion_layers.py` | U-Net layer activations, with the timestep branch marked separately |
| `T19_export_scatter_labels.py` | the latent scatter, with digit labels |
| `T20_export_latent_dims.py` | four VAEs at different latent sizes |
| `T23_export_decoder_layers.py` | decoder activations over a 12×12 grid of latent positions |

## Known approximations

- **Decoder activations depend on z**, so they are precomputed over a 12×12 grid of latent
  positions and a click snaps to the nearest.
- **MNIST-scale models.** Sample quality is modest and visible. The tool is about the concepts,
  not about the behaviour of large generative models.
- **Two PCA components explain less of a 32-dimensional space** than of an 8-dimensional one, so
  the higher-dimensional scatters look more smeared. That is the projection, not the model.

## Status

Built for a course deadline in September 2026 and currently under evaluation with participants.
The interface is not being changed while the study is running.

## References

- Kingma and Welling (2019). *An Introduction to Variational Autoencoders.* arXiv:1906.02691
- Ho, Jain and Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239
- Lee et al. (2024). *Diffusion Explainer.* IEEE VIS. arXiv:2305.03509
- Bertucci and Endert (2024). *VAE Explainer.* arXiv:2409.09011
- Wang et al. (2021). *CNN Explainer.* IEEE TVCG 27(2), 1396–1406
