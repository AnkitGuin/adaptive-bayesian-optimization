"""Page-Hinkley drift-detector unit tests (spec §41)."""

import numpy as np
import pytest

from src.drift import PageHinkley
from src.utils.seeds import rng_for


def _feed(det, values):
    flags = []
    for v in values:
        flags.append(det.update(v))
    return flags


def test_no_false_alarm_on_stable_stream():
    rng = rng_for("td-stable", 0)
    det = PageHinkley(delta=0.5, threshold=25.0)
    # |N(0,1)| residuals, mean ~0.8, std ~0.6 -- a well-calibrated model
    vals = np.abs(rng.normal(0.0, 1.0, size=400))
    flags = _feed(det, vals)
    assert not any(flags), "stable stream must not trigger drift"


def test_detects_shifted_stream_quickly():
    rng = rng_for("td-shift", 1)
    det = PageHinkley(delta=0.5, threshold=25.0)
    stable = np.abs(rng.normal(0.0, 1.0, size=100))
    shifted = np.abs(rng.normal(2.5, 1.0, size=200))  # mean ~3.3
    flags = _feed(det, np.concatenate([stable, shifted]))
    first = next((i for i, f in enumerate(flags) if f), None)
    assert first is not None, "shift must be detected"
    assert first >= 100, "detection must happen after the change"
    assert first - 100 < 60, "detection should be reasonably prompt"


def test_single_event_resets_statistic():
    rng = rng_for("td-reset", 2)
    det = PageHinkley(delta=0.5, threshold=10.0)
    _feed(det, np.abs(rng.normal(0.0, 1.0, size=60)))
    flags = _feed(det, np.abs(rng.normal(3.0, 1.0, size=100)))
    assert any(flags)
    # after the reset, the (now re-baselined) detector is quiet again
    assert not any(flags[30:]), "post-detection residuals should re-baseline"


def test_reset_and_state():
    det = PageHinkley(delta=0.5, threshold=20.0)
    _feed(det, np.abs(np.random.default_rng(3).normal(0, 1, 30)))
    st = det.state()
    assert st["n_updates"] == 30
    det.reset()
    assert det.state()["n_updates"] == 0
    assert det.statistic == 0.0


def test_warmup_does_not_trigger():
    det = PageHinkley(delta=0.0, threshold=0.5)
    # huge values during warm-up (fewer than min_samples in the buffer)
    flags = _feed(det, [50.0, 50.0])
    assert not any(flags)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        PageHinkley(min_samples=1)
    with pytest.raises(ValueError):
        PageHinkley(threshold=0.0)
    with pytest.raises(ValueError):
        PageHinkley(delta=-0.1)
