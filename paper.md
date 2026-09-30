# Two Ways to Make an Image: An Interactive Explainer for the Relationship Between Variational Autoencoders and Diffusion Models

**SaiRishab Kandukuri**
MSc Computer Science, University of Regensburg
*Generative AI for Human-Computer Interaction* — supervised by Prof. Dr.-Ing. Bernd Ludwig

---

## Abstract

Variational autoencoders and denoising diffusion models are usually taught separately, leaving
learners with two disconnected mental models and a set of predictable misconceptions — most
commonly, that a diffusion model encodes into a latent space in the same sense a VAE does. I
present an interactive explainer built around three such misconceptions, showing where the two
architectures genuinely diverge using measurements taken from the trained models themselves:
the VAE encoder reaches an embedding of two values, a 392× reduction from its input, while the
diffusion U-Net's narrowest interior layer still holds four times the image it was given and
never forms an embedding at all. I evaluate it with a pre/post comprehension study reported per
objective, with success criteria and failure conditions fixed in advance. Three of four
participants improved and "I don't know" responses halved, but the item testing the central
misconception was answered incorrectly by every participant at both points, and one moved from
holding no belief to holding the wrong one. I argue that a visualisation whose surface reading
contradicts its intended lesson may reinforce the misconception it targets.

**Live tool:** https://arikaki.github.io/GenAI/

---

# 1 Introduction

Variational autoencoders and denoising diffusion models are both taught as ways of generating
images from noise, and both are usually taught alone. A course covers one, then the other, and a
short section states that the two are related. The relationship is asserted rather than shown.

The consequence is a predictable set of wrong beliefs. The most common, in my experience of this
material, is that a diffusion model "encodes into a latent space" in the same sense a VAE does —
that the noise it starts from is a compressed code. A second is that each reverse step is the
network drawing a slightly better picture. A third is that the two families rest on unrelated
theory: that a VAE is an autoencoder with noise added, and diffusion is iterative denoising, with
no shared foundation.

None of the three is true, and each is testable. That makes them a usable target: an objective
phrased as *understanding the relationship* cannot be assessed, because it cannot be held wrongly.

**This work.** I built an interactive explainer whose subject is not a model but a *relationship*,
and evaluated whether it corrects those three beliefs. The tool presents a VAE and a denoising
diffusion model trained on the same data, opening inside the two networks and then following a
single random draw out through each of them. Every image it shows is a real output of the trained
models. It runs as a static web page with no installation and no server, apart from a small
in-browser encoder used for one interactive feature.

**Contributions.**

1. An interactive explainer targeting three specific, documented-in-advance misconceptions about
   the VAE–diffusion relationship, built from the models' own measured behaviour rather than from
   schematic diagrams.
2. A measurement of where the two architectures actually diverge, taken from the trained networks:
   the VAE encoder reaches an embedding of **2** values, a 392× reduction from its input, while the
   U-Net's narrowest interior layer still holds **4,096** — four times the image it was given — and
   returns to full size, never forming an embedding at all.
3. A didactic evaluation with pre/post comprehension measurement, reported per objective, with
   success criteria and failure conditions fixed before data collection. Three of four
   participants improved and the median score rose from 1 to 5 of 9 — but the item testing the
   primary objective's central misconception was answered incorrectly by every participant at
   both points, and one moved from having no belief to holding the wrong one.

---

---

# 2 Related Work

## 2.1 Interactive explainers for neural architectures

A line of work from the visualization community builds browser-based interactive explainers for
neural network architectures, aimed at learners rather than practitioners. CNN Explainer
(Wang et al., 2021) established the pattern: a model overview integrated with on-demand
explanation views, letting a learner move between the high-level structure of a convolutional
network and the arithmetic of an individual convolution. Its design was informed by interviews
with instructors and a survey of former students, and it is delivered as a web page requiring no
installation.

The same group extended the format to other architectures. Transformer Explainer
(Cho et al., 2025) applies it to text generation, running a live GPT-2 instance in the browser so
that learners can supply their own input and watch how components interact to produce the next
token. Diffusion Explainer (Lee et al., 2024) applies it to text-to-image generation, visualising
how Stable Diffusion transforms a prompt into an image and allowing side-by-side comparison of
generations under different prompts and guidance scales.

Two tools address the autoencoder family directly. VAE Explainer (Bertucci and Endert, 2024)
runs a variational autoencoder live in the browser, adding interaction to the model inputs, the
latent space and the output, and connecting the high-level summary to annotated code and a live
computational graph. It is positioned explicitly as a supplement to existing static
documentation — the Keras code examples in particular — rather than a replacement for them.
The same author has extended the approach to vector-quantized variants in a live tool without an
accompanying paper.[^vqvae]

[^vqvae]: https://xnought.github.io/vq-vae-explainer/

## 2.2 What these tools leave open

Each of these tools explains **one** architecture. That is a reasonable scope decision, and each
executes it well. The consequence, however, is that the relationships *between* model families
are not addressed by any of them.

This matters here specifically. VAE Explainer treats the variational autoencoder as a
self-contained object: an encoder, a latent space, a decoder. Diffusion Explainer treats
diffusion as a self-contained object, and deliberately abstracts away the underlying
probabilistic formulation in order to remain accessible to non-experts with no machine learning
background. Both choices are correct for their stated audiences. But a learner who has worked
through both tools has been given two disconnected mental models, and nothing that connects them.

The connection is not incidental. The denoising diffusion model of Ho et al. (2020) is derived
from a variational bound of exactly the kind that underlies the VAE (Kingma and Welling, 2014,
2019). The two families share an objective; they differ in whether the inference model is learned
or fixed, and in whether the latent variables are lower-dimensional than the data. A learner who
does not see this is likely to form the misconception that the families are unrelated — or, more
commonly in my experience, that diffusion "encodes to a latent space" in the same sense that a
VAE does.

## 2.3 Positioning of this work

I therefore built an interactive explainer whose subject is not a model but a *relationship*. The
tool presents a VAE and a denoising diffusion model side by side and is organised around three
specific misconceptions rather than around an architecture. It differs from the prior tools in
three ways.

**Audience.** Diffusion Explainer and VAE Explainer target non-experts and abstract the
mathematics accordingly. I target learners who have already met both models in a course and need
the probabilistic connection made explicit, so the objective functions are retained rather than
hidden.

**Subject.** The prior tools each explain a single architecture. I explain the correspondence
between two, and the interface is organised around the points of difference — the encoder, the
prediction target, and the shared bound — rather than around a forward pass.

**Evaluation.** I follow the evaluation format of Diffusion Explainer, which reports a
within-subjects user study, but at a considerably smaller scale and with a didactic rather than a
usability focus: I measure change in conceptual understanding against each participant's own
pre-test, and do not measure interface usability.

Like all of these tools, the artifact precomputes model outputs and ships as a static site, so it
requires no installation, no GPU and no server. It is worth noting that CNN, Diffusion and
Transformer Explainer all originate substantially from one research group, so "related work" here
describes a single line of development rather than a broad literature.

# 3 Background

This section fixes notation and states the correspondence the tool is built to teach. Both
formulations are standard; the emphasis on their relationship follows Kingma and Welling (2019)
and the variational derivation in Ho et al. (2020).

## 3.1 The variational autoencoder

A VAE models data $x$ through a latent variable $z$ with prior $p(z) = \mathcal{N}(0, I)$. Two
networks are learned: an inference model (encoder) $q_\phi(z \mid x)$, which outputs the
parameters of a diagonal Gaussian rather than a point, and a generative model (decoder)
$p_\theta(x \mid z)$. Sampling is made differentiable by the reparameterisation trick. Training
maximises the evidence lower bound on $\log p_\theta(x)$:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z \mid x) \,\|\, p(z)\right)}_{\text{regularisation}}$$

Two properties matter for what follows. The inference model is **learned** — $\phi$ is optimised
jointly with $\theta$. And $z$ is typically of much **lower dimension** than $x$; in my
implementation $x \in \mathbb{R}^{784}$ and $z \in \mathbb{R}^{2}$.

## 3.2 Denoising diffusion probabilistic models

A diffusion model defines a fixed forward process that gradually corrupts data over $T$ steps
according to a variance schedule $\beta_1, \dots, \beta_T$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I\right)$$

Writing $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$, this admits a
closed form for any step directly from the data:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t) I\right)
\qquad\Longleftrightarrow\qquad
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$

The generative model reverses this process, $p_\theta(x_{t-1} \mid x_t) =
\mathcal{N}(\mu_\theta(x_t, t), \Sigma_t)$, starting from $x_T \sim \mathcal{N}(0, I)$.

Crucially, the network is not trained to output $\mu_\theta$ or $x_0$ directly. Under the
parameterisation of Ho et al. (2020), it predicts the noise, and the training objective reduces
to a simple regression:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\| \epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon,\; t\right) \right\|^2\right]$$

An estimate of the clean image is then recovered algebraically, by rearranging the closed form
above:

$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\; \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

Note that $\hat{x}_0$ is *computed*, not drawn. The network never emits an image.

## 3.3 The correspondence

The diffusion objective is itself a variational bound. Ho et al. (2020) optimise

$$\mathbb{E}_q\left[\underbrace{D_{\mathrm{KL}}(q(x_T \mid x_0) \| p(x_T))}_{L_T} + \sum_{t>1} \underbrace{D_{\mathrm{KL}}\!\left(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t)\right)}_{L_{t-1}} \underbrace{-\log p_\theta(x_0 \mid x_1)}_{L_0}\right]$$

The structural parallel with $\mathcal{L}_{\text{VAE}}$ is direct: $L_0$ is a reconstruction term,
and the remaining terms are KL divergences between an inference distribution and a learned
generative one. A diffusion model can accordingly be described as a **hierarchical VAE with $T$
layers of latent variables, in which the inference model is fixed rather than learned and the
latent variables have the same dimension as the data**.

This single sentence is the payload of the tool. It also isolates precisely where the two
families diverge, which gives the three learning objectives:

| | VAE | Diffusion |
| --- | --- | --- |
| **Inference model** | learned ($q_\phi$) | fixed ($q$, no parameters) |
| **Latent dimension** | lower than data ($784 \to 2$) | equal to data ($784 \to 784$) |
| **Latent structure** | one step | $T$ steps |
| **Network output** | image pixels | noise $\epsilon$ |
| **Objective** | ELBO | ELBO (decomposed over $t$) |

## 3.4 Concepts targeted by the tool

From the table, three concepts follow, each stated as a misconception the tool is built to
correct:

**C1 — the encoder differs in kind.** Learners familiar with autoencoders tend to read the
forward diffusion process as an encoder in the VAE sense. It is not: it has no learned
parameters, and it does not compress.

**C2 — the network predicts noise, not the image.** The reverse process is commonly read as the
model progressively drawing a better picture. It is not: the network outputs $\epsilon_\theta$,
and $\hat{x}_0$ is recovered by rearrangement.

**C3 — both optimise an ELBO.** The two families are commonly held to rest on unrelated theory.
They do not.

Section 4 describes how each concept is mapped to an interaction, and Section 5 how each is
assessed.

---

---

# 4 System Design

## 4.1 What the interface has to do

Three learning objectives, each defined by the belief it displaces (§2):

**C1** the encoder differs in kind · **C2** the network predicts noise, not the image ·
**C3** both optimise an ELBO-style objective.

Two design rules follow. First, **any element that does not serve one of the three is a candidate
for removal** — stated as a rule because an explainer accretes interesting material otherwise, and
attention is the scarcest resource in a ten-minute session. Second, **claims are measured rather
than asserted wherever the models can supply the measurement.** A caption saying that different
digits occupy different regions is weaker than a readout computing, on demand, that 82% of the
forty nearest encoded images to the chosen point are 3s.

## 4.2 Architecture

Model outputs are precomputed. The VAE and the diffusion model are trained offline; latent
manifolds, reconstructions, forward and reverse trajectories, noise predictions, per-layer
activations and the variance schedule are exported once as sprite sheets and JSON. The page reads
those files and draws to canvas. There is no server, no GPU requirement and no network dependency
at view time.

Sprite geometry is never hard-coded in the front end. Tile size, grid dimensions, station lists
and column counts are all read from the accompanying JSON, so re-exporting with different
settings does not silently break the rendering — a property that mattered in practice, since the
models were re-exported several times during development.

One feature departs from this. Free-hand drawing requires encoding an unseen input, which cannot
be precomputed, so three encoders are bundled as TensorFlow.js models and run in the browser. The
library is bundled locally rather than loaded from a CDN, so the page still works offline.

**Projection.** Embeddings from latent spaces larger than two dimensions are projected to 2-D with
PCA rather than t-SNE. t-SNE has no out-of-sample extension: a newly drawn digit cannot be placed
on an existing t-SNE map without refitting the projection, which would move every other point.
PCA projects a new embedding with a single matrix multiplication.

## 4.3 The sections, in order

The page opens inside the networks and then follows one draw out through each of them, in seven
sections: the layer-by-layer compression comparison; the choice of a starting vector on a map of
the latent space; the decoder expanding that vector into an image; the two-hundred-step reverse
process; a comparison showing where randomness enters each model; the reconstruction and KL terms
over training; and the same architecture trained at four latent sizes. Figures 1–3 show the
three that carry the most argumentative weight.

**How the input is compressed.** Every station is a real layer, probed with a forward pass. Tile
size follows the spatial dimension; bar height follows the number of values held, on a
logarithmic scale shared across both models so the two are directly comparable.

> ![Figure 3 — the two station rows on the shared logarithmic scale.](fig%203.jpeg)
>
> **[FIGURE 3 — the two station rows on the shared logarithmic scale.]**

| | input | peak | narrowest interior | output |
| --- | --- | --- | --- | --- |
| VAE encoder | 784 | 6,272 (8×) | **2** | 2 — an embedding |
| Diffusion U-Net | 1,024 | 65,536 (64×) | **4,096 (4×)** | 1,024 — image-shaped |

Three things visible here are not visible in a block diagram. Compression is **not monotonic**:
the VAE's first convolution increases the representation from 784 to 6,272 values, because the
spatial size halves while the channels grow from 1 to 32; real compression happens later, at the
dense layers. The **U-Net never compresses at all** — its narrowest interior layer holds four
times its input, and the spatial reduction from 32×32 to 4×4 is offset by a rising channel count.
And **only one of the two forms an embedding.** This is C1 in the models' own numbers.

A structural asymmetry surfaced while building this view. The bottleneck cannot be located by
examining spatial layers alone: for the VAE the bottleneck *is* a dense layer — that is what makes
it an encoder — while the U-Net has no dense layer on the image path at all. Its only dense layers
belong to the timestep embedding, a side branch the image never traverses, rendered separately in
the interface for that reason.

> ![Figure 1 — overview of the interactive explainer showing the VAE and diffusion model side by side.](fig%201.png)
>
> **[FIGURE 1 — interface overview of the VAE–diffusion explainer.]**

**One draw, two sizes.** Both models begin by sampling from N(0, I). The tool shows one draw
twice: as two numbers for the VAE, and as a 32×32 field for the diffusion model. They are not the
same vector — they differ in dimension by a factor of 512 — and the interface says so explicitly,
with separate controls to re-roll each. That difference in size is itself C1. The draw is chosen
on a map of 5,000 encoded test images coloured by digit class, and on each selection the tool
reports the composition of the forty nearest. At a cluster core this reaches 100%; between
clusters it falls to around 40% with five digits present. The latent space is **continuous, not
partitioned**, and a learner discovers this by clicking rather than being told.

> ![Figure 2a — x̂₀ selected at t = 190.](fig2a.png)
>
> **[FIGURE 2A — x̂₀ selected at t = 190.]**

**The reverse process** offers a toggle between the noisy state x_t, the network's output ε̂, and
the clean-image estimate x̂₀ recovered from it by rearranging the forward equation. Stepping
through with x̂₀ selected shows a blurred average sharpening into a digit, while the caption states
that the network never draws this image. This is the principal instrument for C2.

**Changing the size of the latent space** trains the same architecture four times.

| latent dim | compression | reconstruction loss |
| --- | --- | --- |
| 1 | 784× | 165.4 |
| **2** | **392×** | **143.6** |
| 8 | 98× | 89.3 |
| 32 | 24.5× | 76.9 |

More dimensions means better reconstruction and less compression; there is no correct answer,
only a trade-off. Two dimensions is used elsewhere on the page for one reason, stated in the
interface: it is the largest latent space that can be *drawn*. One methodological note — the lab
architecture places a 16-unit dense layer before the latent layer, so above 16 dimensions the
effective bottleneck is 16. The comparison models therefore use a wider pre-latent layer and vary
only the latent dimension; that reconstruction loss falls monotonically across all four sizes
confirms the intended behaviour.

## 4.4 Objective to interaction

| Objective | Where the tool teaches it |
| --- | --- |
| **C1** encoder differs in kind | The compression comparison; the two representations of one draw |
| **C2** predicts noise, not image | The x_t / ε̂ / x̂₀ toggle |
| **C3** shared ELBO objective | The reconstruction/KL training curves |
| all three, applied | Where the randomness lives |

**A known gap.** The training-curve section covers the VAE only. Two of the three C3 items concern
what bound *both* models maximise and whether diffusion is a hierarchical VAE — neither is shown
anywhere in the tool, and C3 is consequently supported by the report rather than by the interface.
This is recorded here because it was known before data collection, not discovered afterwards.
In the event C3 *did* move, for two of the three participants who engaged (§6.4), which suggests
the reconstruction-and-KL section carries more of the objective than anticipated — though the
attribution is weaker here than for C1, since the report rather than the interface supplies two
of the three items' content.

## 4.5 Deliberately excluded

Text conditioning and prompt-to-image behaviour; CLIP; real-time inference beyond the single
encoding feature; attention visualisation; and interface usability measures. The first is covered
by existing tools, and duplicating it would consume attention the three objectives need. The last
is excluded because a usability evaluation lies outside this module's learning objectives.

# 5 Study Design

## 5.1 The claim

That a learner holding a specific incorrect model of the relationship holds a less incorrect one
after using the tool. Not that the interface is pleasant, fast or preferred: a usability
evaluation is outside the module's learning objectives, and measuring it would consume participant
time the comprehension measure needs.

## 5.2 Misconception-based items

Each objective is defined by the false belief it displaces. This makes items **diagnostic** rather
than merely difficult — a wrong answer names which misconception survived — and makes the
objectives falsifiable, since each is a proposition with a truth value. The cost is narrowness:
the instrument measures three propositions, not comprehension of generative modelling.

The approach follows the concept-inventory tradition in physics education, where instruments are
built around documented misconceptions rather than curriculum coverage (Hestenes et al., 1992).
The misconceptions here are *not* independently documented; they are derived from the lecture
material and from the structure of the two model families, which is a weaker foundation and is
listed among the limitations.

## 5.3 Each participant as their own baseline

A between-subjects comparison against a second teaching condition was rejected on three grounds.
It cannot be powered at the achievable sample size; recruitment could not supply the numbers,
since the study window fell in the lecture-free period; and the comparison asks a question about
media, whereas the module assesses didactics.

**What this costs.** There is no counterfactual. A pre/post gain cannot be attributed to the tool
as against any comparable period spent thinking about the topic. The session is short and
continuous, and participants are asked whether they consulted other material, but neither measure
eliminates the problem. This is the largest limitation of the design.

## 5.4 Identical items, and the recall confound

Repeating the nine items verbatim makes each participant's change directly interpretable, with no
equating between forms required; constructing and validating a matched second form is not
feasible at this sample size. The cost is recall — a participant may improve because they
remember being asked.

**Two transfer items** administered post-only address this. They probe the same objectives in a
form not seen beforehand and require a written justification rather than a selection. If scores
rise on the repeated items while transfer answers are poor, that pattern indicates recall rather
than understanding, and is reported as such.

## 5.5 Two features of the instrument, one of them missing

**"I don't know" appears on every knowledge item.** Without it a participant holding no belief
guesses and scores at chance, inflating the pre-test and conflating a wrong belief with no belief.
It also yields a second signal at no cost: a fall in "I don't know" responses is evidence of
learning even where the answer remains wrong. This proved the most informative measure in the
study (§6.3).

**A confidence rating was specified but not implemented.** The design called for a 1–5 rating
after each item, because a binary score conflates knowing, not knowing, and being confidently
wrong — and only the third is a misconception in the relevant sense. The delivered instrument
contains no such ratings; the omission was discovered at analysis. Confidence calibration is
therefore not reported, and one of the three failure conditions fixed in §5.8 — a rise in
confidence without a rise in accuracy — **cannot be assessed at all**. A pre-registered check that
was never measurable is a defect in execution, not a neutral omission.

## 5.6 Procedure

Participants receive a written information sheet and sign a consent form before the session. Each
is assigned a participant code; responses are recorded under that code only, and the signed forms
are held separately on paper, so the data is pseudonymised rather than anonymous. No name, email
address or other identifier is collected in the instrument.

A session takes approximately 25 minutes and is completed online, unsupervised, at a time of the
participant's choosing: background questions, nine pre-test items, roughly ten minutes with the
tool, the same nine items again, and two transfer items with written justifications. Participants are asked not to prepare beforehand and not to consult other material
during the session; a final item records whether they did.

The instrument was delivered with Microsoft Forms through the university's environment. No
question other than the consent confirmation, the participant code and the final
other-material question is compulsory, consistent with the undertaking in the consent sheet that
any item may be declined.

## 5.7 Participants

Eligibility is prior exposure to machine learning rather than attendance of this specific module:
a participant with no such background has no misconception to correct, only an absence, and
cannot exhibit the effect the tool is built to produce. Background items record degree programme,
whether the participant took this module, whether they have completed any machine learning course,
and their prior exposure to each model family. No participant is excluded on these answers; the
composition is reported.

**Achieved sample: four**, against a target of eight to twelve, collected between 5 and 28
September 2026. All four had completed a machine learning course. Three were MSc Computer Science
students who had attended the module; one was not currently studying and had not. One reported
having used or implemented both model families; the other three had heard of them without using
them. A fifth submission, from the supervisor verifying that the instrument was technically sound,
answered "I don't know" throughout so it could be identified, and is excluded.

## 5.8 Scoring and criteria, fixed in advance

One point per correct core item (maximum 9), with subscores per objective (maximum 3 each).
Transfer items score one point each, requiring both the correct answer and a reason naming the
mechanism. Skipped items score as **missing, not wrong** — a participant who declined has not
demonstrated a false belief.

**Primary criterion:** the median post-test total exceeds the median pre-test total, and at least
two thirds of participants improve their own total.

**Per objective:** C1, C2 and C3 reported separately. **C2 is the primary objective**; a gain on
C2 with C1 and C3 flat remains a meaningful result.

**No significance testing.** At this sample size a p-value would invite a claim about a population
the sample cannot support. The study is formative: it reports direction, the pattern across
objectives, confidence calibration, and participants' own explanations.

**Three outcomes counting against the tool**, stated before collection. No movement. Movement on
repeated items but not transfer items, indicating recall. And a rise in confidence without a rise
in accuracy — worse than no learning, since a confidently wrong learner is harder to correct than
an uncertain one. The third could not be evaluated, for the reason given in §5.5.

---

---

# 6 Results

## 6.1 Participants

Four participants completed the study between 5 and 28 September 2026, against a target of eight
to twelve. All four had completed a machine learning course. Three were MSc Computer Science
students who had attended the module; one was not currently studying and had not. One reported
having used or implemented both model families; the other three had heard of them without using
them.

The shortfall against target is the dominant constraint on everything below. With four
participants — one of whom did not engage with the tool (§6.2) — the study reports the direction
and shape of the change it observed, and no more than that.

## 6.2 One response reflects no exposure to the tool

**P05 completed the entire protocol in 2.2 minutes**, against a design duration of approximately
25 minutes including around ten minutes with the tool. Their score was 0 of 9 at both points,
their "I don't know" count *rose* from 7 to 9, and both transfer items were left unjustified. The
most parsimonious reading is that the tool was not used.

The response is retained in the reported sample and shown in all tables — excluding it on a
criterion formed after seeing the data would be worse — but it is excluded from statements about
change, since a participant who did not receive the intervention cannot evidence its effect.
Where a figure is given for "participants who engaged", it refers to P01, P02 and P03, whose
sessions ran 28.4, 19.0 and 16.9 minutes.

## 6.3 Core comprehension

| | pre | post | change | "don't know" pre → post | session |
| --- | --- | --- | --- | --- | --- |
| P01 | 1 | 3 | +2 | 8 → 2 | 28.4 min |
| P02 | 5 | 7 | +2 | 1 → 0 | 19.0 min |
| P03 | 1 | **8** | **+7** | 6 → 0 | 16.9 min |
| P05 | 0 | 0 | 0 | 7 → 9 | 2.2 min |

> **[FIGURE 4 — slopegraph: pre to post total for each of the four participants.]**

Median pre-test 1.0, median post-test 5.0. Three of four participants improved; none declined.
Among the three who engaged, all three improved, and the median rose from 1 to 7.

**The pre-registered primary criterion is met**: the median post-test total exceeds the median
pre-test total, and three of four — two thirds or more — improved their own score. This is stated
with the sample size firmly in view; at n = 4 the criterion is easily satisfied and carries
correspondingly little weight.

**"I don't know" responses fell from 22 to 11 across the sample**, and from 15 to 2 among the
three who engaged. P03 and P02 finished with none at all. Since the option was offered explicitly
and participants were instructed to prefer it to guessing, this is the clearest signal in the
data: the tool moved participants out of "no belief" and into holding a position. Whether the
position was correct is the subject of §6.4.

## 6.4 Per objective

Maximum 3 per objective.

| | C1 pre → post | C2 pre → post | C3 pre → post |
| --- | --- | --- | --- |
| P01 | 1 → 2 | 0 → 0 | 0 → 1 |
| P02 | 2 → 2 | 0 → 2 | 3 → 3 |
| P03 | 0 → 3 | 1 → 2 | 0 → 3 |
| P05 | 0 → 0 | 0 → 0 | 0 → 0 |

**C1 (the encoder differs in kind)** improved for two of three engaged participants, and P03 moved
from 0 to full marks. This is the objective the reordered opening section addresses most directly.

**C3 (both optimise an ELBO)** improved for two of three, including P03 from 0 to 3. This is
notable because §4.4 recorded in advance that the interface supports C3 only partially — two of
its three items concern material shown in the report but not in the tool. The gain here is
therefore harder to attribute to the interface than the C1 gain is, and may reflect the
reconstruction-and-KL section carrying more of C3 than anticipated.

**C2 (the network predicts noise, not the image)** is the primary objective, and it produced the
study's most important finding.

## 6.5 Item 6: the central misconception was not corrected

Item 6 states: *"At each reverse step, the network directly draws a slightly cleaner version of
the image."* The correct answer is False. It targets precisely the belief the tool's x̂₀ toggle was
built to displace.

**No participant answered it correctly, before or after.**

| | pre | post |
| --- | --- | --- |
| P01 | I don't know | **True** |
| P02 | True | True |
| P03 | True | True |
| P05 | I don't know | I don't know |

P02 and P03 held the misconception before and retained it after. P01 moved from *no belief* to
*the wrong belief* — an outcome worse than no change, and one of the three failure conditions
fixed in advance in §5.8.

This is a negative result on the objective designated primary, and it is not softened by the gains
elsewhere. Two readings are available and the data cannot separate them:

1. **The x̂₀ view does not communicate what it was designed to.** A learner can watch a blurred
   average sharpen into a digit and take that as confirmation that the model is progressively
   drawing the picture — the opposite of the intended lesson. The caption states that the network
   never draws this image, but a caption competing with a compelling animation is a weak
   instrument.
2. **The item may be misread.** "Directly draws" is doing load-bearing work in the wording, and a
   participant who understands ε-prediction perfectly might still read the statement as a loose
   description of what the reverse process achieves.

Distinguishing the two requires a think-aloud protocol or a revised item, neither available here.
That the other two C2 items *did* improve for P02 and P03 weakly favours the second reading — a
participant who learned nothing about noise prediction would be unlikely to gain on items 4 and 5
— but this cannot be settled at n = 3.

## 6.6 Transfer items

Scored by the pre-registered rubric: one point requires the correct answer *and* a reason naming
the mechanism.

| | same z twice? | justification | same image? | justification |
| --- | --- | --- | --- | --- |
| P01 | No ✓ | "random inits" — does not name sampling | **Yes ✗** | "No because of different Noise" |
| P02 | No ✓ | "random seed + sampling" ✓ | No ✓ | "changes compound" — does not name the initial noise |
| P03 | No ✓ | "does not always directly output a fixed vector" ✓ | No ✓ | "different random noise… so you cannot get the same image" ✓ |
| P05 | I don't know | — | I don't know | — |

Two judgement calls are recorded rather than hidden. **P01's second response contradicts itself**
— the selection is "Yes" while the justification reads "No because of different Noise" — and is
scored 0 under the rubric, though the reasoning is correct and the discrepancy is most likely a
mis-click. **P02's "changes compound"** describes divergence without naming the starting noise as
its cause, and is scored 0 on a strict reading. Under the strict rubric: P03 scores 2, P02 scores
1, P01 and P05 score 0.

The three who engaged all answered the first transfer item correctly, which is the item testing
the stochastic encoder — content the "where the randomness lives" section shows directly. Since
neither transfer item appeared in the pre-test, correct answers here cannot be explained by recall
of the earlier questions, and this is the strongest available evidence that the change on C1 is
understanding rather than memorisation.

## 6.7 Confounding exposure

**Two of four participants — P02 and P03 — reported consulting other material during the session.**
These are the two highest post-test scores and include the largest single gain.

This substantially weakens the attribution argument already flagged as the design's principal
limitation (§5.3). With no control condition, the final item asking about outside material was the
only mitigation available, and at this sample size it reports that half the participants did what
they were asked not to. P03's +7 gain, the study's headline number, is among the affected
responses.

The honest position is that the study cannot separate the tool's contribution from that of the
material participants consulted alongside it.

## 6.8 Interface change during the collection period

The interface was substantially reordered on **18 September 2026** following supervisor feedback,
moving the layer-by-layer comparison to the opening position and adding explicit model labelling
throughout. Sessions took place on 5, 10, 19 and 28 September, so the sample divides evenly:
**P02 and P01 used the original arrangement; P03 and P05 used the revised one.**

Two observations follow, and neither can be pressed. The two participants on the original
arrangement each gained 2 points. Of the two on the revised one, P03 gained 7 — the largest gain
in the study — and P05 did not engage with the tool at all. Among participants who actually used
it, that leaves two on the earlier version and one on the later, which cannot support a
comparison and none is attempted. The change is recorded because it bears on the interpretation
of any pattern in the data, and because concealing it would misrepresent what participants saw.

## 6.9 Open responses

Three participants answered the free-text questions; with three responses no coding is possible
and they are reported directly.

Asked what became clearer, participants named the three-seeds comparison specifically, the
comparisons and visualisations generally, and the working of both model families.

Asked what was confusing, the three criticisms did not overlap. The most substantive came from
the participant with the highest prior exposure: *"KL is not gone deep into enough detail
(mathematical nor intuitive), and the statistical side (sampling, etc) is not touched"* (P02).
This is a fair description of a deliberate scope decision, and it points at C3 — the objective is
supported by one section, and that section is descriptive rather than derivational. P01 reported
that the *"structure of content was confusing"*, recorded eight days before the reordering
intended to address exactly that (§6.8); whether the change resolves it is untested. P03 asked
for a brief background on why these particular examples were chosen, which the tool never
supplies.

## 6.10 Summary against the pre-registered criteria

| Criterion (fixed in §5.8) | Outcome |
| --- | --- |
| Median post exceeds median pre | **Met** — 1.0 → 5.0 |
| At least two thirds improve | **Met** — 3 of 4 |
| C2 (primary objective) | **Mixed**, and failed on its central item |
| Failure: no movement | Not triggered |
| Failure: repeated-item gain without transfer | Not triggered — transfer performance tracks core gains |
| Failure: confidence up, accuracy flat | **Not assessable** — see §5 correction |

---

---

# 7 Discussion and Limitations

§6 reports the outcome; this section interprets it. C1 moved for two of the three participants
who engaged with the tool, and C3 for two — but the item testing the central misconception behind
C2, the primary objective, was answered incorrectly by every participant at both measurement
points. The gains are real and the failure is unambiguous, and the limitations below bear on how
much weight either can carry.

## Limitations

1. **No control condition.** A gain cannot be attributed to the tool as against any comparable
   period of engagement with the topic. The most significant limitation of the design (§5.3).
2. **Half the sample consulted outside material** during the session despite being asked not to,
   including the two largest gains (§6.7). With no control condition the final questionnaire item
   was the only available mitigation, and it reports a problem rather than reassurance.
3. **Four participants against a target of eight to twelve**, recruited during the lecture-free
   period despite three rounds of contact with students and faculty. The pre-registered criteria
   are met, but at n = 4 they are satisfied by very few observations and should not be read as
   evidence of an effect in any population. Participants were also self-selected and are likely
   more motivated than typical learners.
4. **One participant did not engage with the intervention.** P05 completed the protocol in 2.2
   minutes and scored 0 at both points; retained in the sample but excluded from statements about
   change (§6.2).
5. **No pilot was run** before the study opened, so ceiling and floor effects were not screened in
   advance. One was subsequently found: item 6 was answered incorrectly by every participant at
   both points. A pilot with three participants would very likely have surfaced this while the
   item could still have been revised, and its absence is the most consequential procedural
   shortcut taken here.
6. **Item 6 may be misworded.** "Directly draws" is doing load-bearing work, and a participant who
   understands ε-prediction might still read the statement as a loose description of the reverse
   process. Whether the tool or the item failed cannot be determined from these data.
7. **The interface was modified during the study period**, deployed 18 September 2026, with two
   participants either side (§6.8). No comparison by version is possible at this n.
8. **The confidence measure specified in the design was absent from the delivered instrument**
   (§5.5), so one pre-registered failure condition could not be assessed.
9. **The misconceptions are not independently validated**, and C3 is under-supported by the
   interface (§4.4). Two further points of detail: one transfer response was internally
   inconsistent and scored 0 under the rubric (§6.6), and one submission — the supervisor's
   verification of the instrument — was excluded.

Findings concern comprehension of the concepts at MNIST scale, not the behaviour of large
generative models.

## Future work

**Resolve item 6.** Whether the x̂₀ view fails or the item does is the single most valuable thing a
follow-up could establish, since it determines whether the tool's central interaction works. A
think-aloud session with three or four participants would likely settle it, at far less cost than
another round of collection.

**Address the surface-reading problem.** If the x̂₀ animation reads as *"the model is drawing the
picture"* regardless of its caption, the remedy is structural rather than textual — showing ε̂ and
x̂₀ together, so the algebraic relation between them is visible, rather than as alternatives behind
a toggle where only one is on screen at a time.

Beyond that, two extensions would strengthen the work rather than repair it: an instrument built
on misconceptions established empirically by interviewing learners, and a between-subjects
comparison against a static presentation, which would supply the counterfactual this study lacks.
A third was scoped out during development — projecting the diffusion model's internal states with
PCA to show trajectories converging over timesteps, which would give the diffusion side a spatial
account comparable to the VAE's latent map.

# 8 Conclusion

The objective was reached **gradually, not completely** — and the qualification matters more than
the headline.

Three of four participants improved, the median total rose from 1 to 5, and "don't know"
responses halved. Among the three who engaged with the tool, every one improved and the median
rose from 1 to 7. The transfer items, which no participant had seen before, were answered
correctly by all three, which is the strongest evidence available that the change reflects
understanding rather than recall of the repeated questions. On C1 — that the VAE encoder and the
diffusion forward process differ in kind — the tool appears to do what it was built to do.

Against that: the primary objective's central item was not corrected in a single case, and one
participant moved from having no belief to holding the wrong one. If the aim was to displace the
belief that each reverse step draws a slightly better picture, the evidence here is that it was
not displaced. Half the sample consulted outside material, so the gains that did occur cannot be
cleanly attributed. The sample reached four rather than the eight to twelve intended, one of whom
did not use the tool. The interface changed partway through collection.

The appropriate conclusion is therefore narrow. **The tool moved participants from not knowing to
holding positions, and several of those positions were correct** — the fall in "don't know"
responses from 15 to 2 among engaged participants is the most robust thing in the data. Whether it
corrects the specific misconception it was designed around remains unanswered, and the one item
addressing that directly suggests it does not.

For the didactic question the project set out to ask — whether making the VAE–diffusion
relationship *visible* changes what learners believe about it — the answer this study supports is
that visibility produces engagement and partial conceptual movement, but that a visualisation
whose surface reading contradicts its intended lesson may reinforce the misconception it targets.
The x̂₀ view shows a blurred image becoming sharp. That is exactly what someone who believes the
model progressively draws the picture expects to see. A caption stating otherwise appears not to
be enough.

That is a finding worth having, and it is more useful than the gains would have been alone.

---

# References

Donald Bertucci and Alex Endert. 2024. VAE Explainer: Supplement learning variational
autoencoders with interactive visualization. *arXiv preprint* arXiv:2409.09011.

Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Zijie J. Wang, Seongmin Lee,
Benjamin Hoover, Minsuk Kahng, and Duen Horng Chau. 2025. Transformer Explainer: Interactive
learning of text-generative models. In *Proceedings of the AAAI Conference on Artificial
Intelligence*, 39(28):29625–29627.

David Hestenes, Malcolm Wells, and Gregg Swackhamer. 1992. Force Concept Inventory. *The Physics
Teacher*, 30(3):141–158.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In
*Advances in Neural Information Processing Systems*, volume 33, pages 6840–6851. Curran
Associates, Inc.

Diederik P. Kingma and Max Welling. 2014. Auto-encoding variational Bayes. In *2nd International
Conference on Learning Representations (ICLR)*, Banff, AB, Canada.

Diederik P. Kingma and Max Welling. 2019. An introduction to variational autoencoders.
*Foundations and Trends in Machine Learning*, 12(4):307–392.

Seongmin Lee, Benjamin Hoover, Hendrik Strobelt, Zijie J. Wang, ShengYun Peng, Austin Wright,
Kevin Li, Haekyu Park, Haoyang Yang, and Duen Horng Chau. 2024. Diffusion Explainer: Visual
explanation for text-to-image Stable Diffusion. In *IEEE Visualization and Visual Analytics
(VIS), Short Papers*.

Zijie J. Wang, Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, Fred Hohman, Minsuk Kahng,
and Duen Horng Chau. 2021. CNN Explainer: Learning convolutional neural networks with interactive
visualization. *IEEE Transactions on Visualization and Computer Graphics*, 27(2):1396–1406.