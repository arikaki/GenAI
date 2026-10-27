"""The diffusion maths, checked without a trained model.

The key test is the oracle: if the network predicted the true noise exactly,
the reverse step must equal the true posterior mean q(x_{t-1} | x_t, x0), and
running every step from pure noise must land exactly on x0.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import diffusion_schedule as ds


@pytest.fixture
def s():
    return ds.make_schedule()


@pytest.fixture
def x0():
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(size=(4, 32, 32, 1)), -1, 1)


def true_eps(s, x_t, t, x0):
    """The noise a perfect network would predict, given the clean image."""
    return ((x_t - ds._at(s["sqrt_alpha_bar"], t, x_t.ndim) * x0) /
            ds._at(s["sqrt_one_minus_abar"], t, x_t.ndim))


def test_schedule_ends_at_pure_noise(s):
    assert s["alpha_bar"][-1] < 1e-4
    assert np.all(np.diff(s["alpha_bar"]) < 0)


def test_every_index_carries_noise(s):
    assert s["alpha_bar"][0] < 1


def test_legacy_schedule_did_not_end_at_noise():
    legacy = ds.legacy_schedule()
    assert legacy["alpha_bar"][0] == 1             # index 0 was noise-free
    assert legacy["alpha_bar"][-1] > 0.1           # 36% of the amplitude survived


def test_forward_and_x0_estimate_invert(s, x0):
    rng = np.random.default_rng(1)
    t = np.array([0, 50, 120, 199])
    eps = rng.normal(size=x0.shape)
    x_t = ds.forward_noise(s, x0, t, eps)
    np.testing.assert_allclose(ds.predict_x0(s, x_t, t, eps), x0, atol=1e-6)


@pytest.mark.parametrize("t", [1, 2, 50, 100, 199])
def test_reverse_step_matches_true_posterior(s, x0, t):
    rng = np.random.default_rng(t)
    tt = np.full(len(x0), t)
    x_t = ds.forward_noise(s, x0, tt, rng.normal(size=x0.shape))
    eps = true_eps(s, x_t, tt, x0)
    np.testing.assert_allclose(ds.reverse_mean(s, x_t, tt, eps),
                               ds.posterior_mean(s, x_t, tt, x0), atol=1e-9)


def test_legacy_reverse_step_did_not_match():
    """The notebook stepped with alpha[t] between noise levels that alpha[t-1]
    separates. The mismatch is real, though small."""
    s = ds.legacy_schedule()
    rng = np.random.default_rng(0)
    x0 = np.clip(rng.normal(size=(2, 8, 8, 1)), -1, 1)
    t = 100
    tt = np.full(2, t)
    x_t = ds.forward_noise(s, x0, tt, rng.normal(size=x0.shape))
    eps = true_eps(s, x_t, tt, x0)
    # the posterior between legacy levels t and t-1 uses the step alpha[t-1]
    shifted = dict(s, alpha=np.roll(s["alpha"], 1), beta=np.roll(s["beta"], 1),
                   alpha_bar=np.roll(s["alpha_bar"], -1))
    gap = np.abs(ds.reverse_mean(s, x_t, tt, eps) - ds.posterior_mean(shifted, x_t, tt, x0))
    assert 1e-7 < gap.max() < 1e-2


def test_oracle_sampler_recovers_x0(s, x0):
    rng = np.random.default_rng(2)
    x = rng.normal(size=x0.shape)
    for t in reversed(range(s["T"])):
        eps = true_eps(s, x, np.full(len(x0), t), x0)
        x = ds.reverse_step(s, x, t, eps, rng)
    np.testing.assert_allclose(x, x0, atol=1e-6)


def test_bound_weights_favour_steps_near_the_image(s):
    t, lam = ds.bound_weights(s)
    assert t[0] == 1 and len(t) == s["T"] - 1
    assert np.all(lam > 0)
    assert np.argmax(lam) == 0                      # heaviest at the step nearest the image
    assert lam[0] > 3 * lam[-1]
