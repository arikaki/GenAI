"""The diffusion model's noise schedule, forward process and reverse step.

NumPy only, so the maths can be unit-tested without TensorFlow (tests/) and
shared by the training script (T32) and its exports.

Indexing: array index t = 0 .. T-1 is the paper's timestep t + 1 (Ho et al.
2020). alpha_bar[t] is the product of alpha[0..t], so every index, including
t = 0, carries some noise, and the reverse step from index t uses beta[t] and
alpha[t], the same step the forward process took to reach it.

The lab notebook this replaces differed in three ways:
  1. beta ran linearly from 1e-4 to 0.02 over 200 steps. Those values were
     chosen for T = 1000; over 200 steps they leave alpha_bar_T = 0.13, so
     36% of the image's amplitude survived to the last step, while sampling
     started from pure N(0, I). Here the same linear schedule is rescaled by
     1000 / T, which reaches alpha_bar_T = 3e-5 at T = 200.
  2. alpha_bar was shifted by one (alpha_bar[0] = 1), so index 0 had no noise
     at all and the network was trained there on a target it could not infer.
  3. The reverse step at index t used alpha[t], while the forward step that
     led to that noise level was alpha[t-1]. The two differ by at most 0.01%
     relative, so this was a correctness fix, not a visible one.
legacy_schedule() keeps the old arrays so the tests can show the difference.
"""
import numpy as np

T = 200


def make_schedule(T=T, beta_start=1e-4, beta_end=0.02):
    """Linear beta, rescaled from its T = 1000 values to this T."""
    scale = 1000 / T
    beta = np.linspace(beta_start * scale, beta_end * scale, T, dtype=np.float64)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    return {"T": T, "beta": beta, "alpha": alpha, "alpha_bar": alpha_bar,
            "sqrt_alpha_bar": np.sqrt(alpha_bar),
            "sqrt_one_minus_abar": np.sqrt(1 - alpha_bar)}


def legacy_schedule(T=T):
    """The lab notebook's arrays, kept for comparison only."""
    beta = np.linspace(0.0001, 0.02, T)
    alpha = 1 - beta
    alpha_bar = np.concatenate(([1.], np.cumprod(alpha)[:-1]))
    return {"T": T, "beta": beta, "alpha": alpha, "alpha_bar": alpha_bar,
            "sqrt_alpha_bar": np.sqrt(alpha_bar),
            "sqrt_one_minus_abar": np.sqrt(1 - alpha_bar)}


def _at(arr, t, ndim):
    """Gather per-sample values and shape them to broadcast over an image batch."""
    return np.reshape(np.take(arr, t), (-1,) + (1,) * (ndim - 1))


def forward_noise(s, x0, t, eps):
    """x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps."""
    return (_at(s["sqrt_alpha_bar"], t, x0.ndim) * x0 +
            _at(s["sqrt_one_minus_abar"], t, x0.ndim) * eps)


def predict_x0(s, x_t, t, eps_hat):
    """The clean image implied by a noise prediction: the forward equation rearranged."""
    return ((x_t - _at(s["sqrt_one_minus_abar"], t, x_t.ndim) * eps_hat) /
            _at(s["sqrt_alpha_bar"], t, x_t.ndim))


def reverse_mean(s, x_t, t, eps_hat):
    """Mean of p(x_{t-1} | x_t) under the epsilon parameterisation."""
    a = _at(s["alpha"], t, x_t.ndim)
    b = _at(s["beta"], t, x_t.ndim)
    return (x_t - b / _at(s["sqrt_one_minus_abar"], t, x_t.ndim) * eps_hat) / np.sqrt(a)


def posterior_mean(s, x_t, t, x0):
    """Mean of the true posterior q(x_{t-1} | x_t, x0); alpha_bar before index 0 is 1."""
    ab = _at(s["alpha_bar"], t, x_t.ndim)
    ab_prev = _at(np.concatenate(([1.], s["alpha_bar"][:-1])), t, x_t.ndim)
    a = _at(s["alpha"], t, x_t.ndim)
    b = _at(s["beta"], t, x_t.ndim)
    return (np.sqrt(ab_prev) * b / (1 - ab) * x0 +
            np.sqrt(a) * (1 - ab_prev) / (1 - ab) * x_t)


def reverse_step(s, x_t, t, eps_hat, rng):
    """One ancestral sampling step, sigma_t^2 = beta_t; the last step adds no noise.
    `t` is a single integer index here, as sampling runs one timestep at a time."""
    mean = reverse_mean(s, x_t, t, eps_hat)
    if t == 0:
        return mean
    return mean + np.sqrt(s["beta"][t]) * rng.standard_normal(x_t.shape)


def bound_weights(s):
    """Weight the variational bound puts on each step's squared noise error
    (Ho et al. 2020, eq. 12, with sigma_t^2 = beta_t), for t >= 1; index 0 is
    the reconstruction term, not a KL term."""
    t = np.arange(1, s["T"])
    return t, s["beta"][t] / (2 * s["alpha"][t] * (1 - s["alpha_bar"][t]))
