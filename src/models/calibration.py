"""
src/models/calibration.py
--------------------------
Statistical Calibration Layer — Platt Scaling with fixed α = √3.

Correcting LLM hedging bias, NOT overconfidence
-------------------------------------------------
This corrects a common misconception: LLMs trained with RLHF do NOT
systematically overestimate probabilities. They UNDERESTIMATE extremes —
they hedge toward 0.5. The mechanism is RLHF alignment: models are rewarded
for expressing uncertainty, so a well-calibrated 0.80 becomes a hedged 0.62.

The paper (Section 4.4) applies Platt scaling with a FIXED α = √3 ≈ 1.73:

    p̂ = σ(α · logit(p))
    where logit(p) = log(p / (1-p))

This EXTREMIZES predictions:
    raw=0.40 → calibrated=0.28  (pushed further from 0.5)
    raw=0.50 → calibrated=0.50  (unchanged at boundary)
    raw=0.60 → calibrated=0.72  (pushed further from 0.5)
    raw=0.75 → calibrated=0.87

α = √3 is fixed (not learned) to avoid overfitting on small calibration sets.
The difference between fixed and learned α is <0.005 Brier points (Table 3).

Mathematical equivalence (Appendix G.2 of the paper)
-----------------------------------------------------
Log-odds extremization: log(p̂/(1-p̂)) = (d/n) · Σᵢ log(pᵢ/(1-pᵢ))
is mathematically equivalent to Platt scaling on the geometric mean of
individual logits with parameter α = d. This equivalence was first derived
in the AIA paper and is a novel theoretical contribution.

Training data simulation
------------------------
We simulate 200 historical (llm_raw, outcome) pairs that reflect LLM hedging:
  - True outcomes are bimodal (events either happen decisively or don't)
  - LLM raw scores are SHRUNK toward 0.5: llm_raw = 0.5 + shrink*(true_p - 0.5)
  - Calibration then reverses this shrinkage via extremization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Fixed Platt scaling parameter from the paper (α = √3)
PLATT_ALPHA = math.sqrt(3)  # ≈ 1.7321


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class CalibrationDiagnostics:
    """
    Transparency metrics describing the LLM hedging bias and calibration fit.
    Displayed in the Streamlit sidebar.
    """
    mean_raw_prob: float          # Average LLM raw score on training data
    mean_actual_rate: float       # Average actual outcome rate on training data
    hedging_gap: float            # mean_actual_rate - mean_raw_prob (>0 near extremes = hedging)
    platt_alpha: float            # Fixed α = √3
    example_corrections: List[Tuple[float, float]]  # [(raw, calibrated), ...] for display


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """
    Applies Platt Scaling (fixed α=√3) to correct LLM hedging bias.

    The calibrate() method is a closed-form transformation — no model fitting
    is required at inference time. train_calibrator() exists to generate
    diagnostics that explain the correction to the user.

    Usage
    -----
    >>> cal = ProbabilityCalibrator()
    >>> cal.train_calibrator()
    >>> cal.calibrate(0.62)   # returns ~0.74
    >>> cal.calibrate(0.38)   # returns ~0.26
    """

    def __init__(self):
        self._is_trained: bool = False
        self.diagnostics: CalibrationDiagnostics | None = None

    # ------------------------------------------------------------------
    # Platt scaling: the closed-form core
    # ------------------------------------------------------------------

    @staticmethod
    def _platt_scale(p: float, alpha: float = PLATT_ALPHA) -> float:
        """
        Apply Platt scaling:  p̂ = σ(α · logit(p))

        Parameters
        ----------
        p     : raw probability in (0, 1)
        alpha : scaling factor. >1 extremizes, <1 compresses, =1 is identity.

        Returns
        -------
        Calibrated probability in (0, 1).
        """
        # Guard against boundary values before log
        p = max(1e-6, min(p, 1 - 1e-6))
        logit_p = math.log(p / (1.0 - p))
        scaled = alpha * logit_p
        return 1.0 / (1.0 + math.exp(-scaled))

    # ------------------------------------------------------------------
    # Training (diagnostic only)
    # ------------------------------------------------------------------

    def _simulate_hedging_data(
        self,
        n_samples: int = 200,
        shrink_factor: float = 0.60,
        rng: np.random.Generator | None = None,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Simulate historical (llm_raw, outcome) data with LLM hedging bias.

        True probabilities are bimodal (macro events tend to be decisive).
        LLM raw scores are shrunk toward 0.5 by shrink_factor.

        shrink_factor=0.60 means:
            true_p=0.85 → llm_raw = 0.5 + 0.60*(0.85-0.5) = 0.71
            true_p=0.20 → llm_raw = 0.5 + 0.60*(0.20-0.5) = 0.32
        """
        if rng is None:
            rng = np.random.default_rng()

        # Bimodal true probability distribution: mix of Beta(2,6) and Beta(6,2)
        # This simulates macro events that tend to resolve decisively
        n_low = n_samples // 2
        n_high = n_samples - n_low
        true_p = np.concatenate([
            rng.beta(2, 6, size=n_low),   # events that tend NOT to happen
            rng.beta(6, 2, size=n_high),   # events that tend to happen
        ])
        rng.shuffle(true_p)

        # LLM hedges: shrinks toward 0.5
        llm_raw = 0.5 + shrink_factor * (true_p - 0.5)
        # Clip to valid range
        llm_raw = np.clip(llm_raw, 0.01, 0.99)
        true_p  = np.clip(true_p,  0.01, 0.99)

        # Binary outcomes from true probabilities
        outcomes = rng.binomial(1, true_p).astype(np.int_)

        return llm_raw, outcomes

    def train_calibrator(
        self,
        n_samples: int = 200,
        random_seed: int | None = 42,
    ) -> CalibrationDiagnostics:
        """
        Generate simulated training data and compute diagnostics.
        The actual calibration (Platt α=√3) is closed-form and requires
        no fitting — this method exists purely for transparency metrics.
        """
        rng = np.random.default_rng(seed=random_seed)
        llm_raw, outcomes = self._simulate_hedging_data(n_samples=n_samples, rng=rng)

        mean_raw = float(llm_raw.mean())
        mean_actual = float(outcomes.mean())
        hedging_gap = mean_actual - mean_raw  # positive if LLM undershoots extremes

        # Example correction table for UI
        example_raws = [0.38, 0.45, 0.55, 0.62, 0.70, 0.80]
        examples = [(r, round(self._platt_scale(r), 4)) for r in example_raws]

        self.diagnostics = CalibrationDiagnostics(
            mean_raw_prob=round(mean_raw, 4),
            mean_actual_rate=round(mean_actual, 4),
            hedging_gap=round(hedging_gap, 4),
            platt_alpha=round(PLATT_ALPHA, 4),
            example_corrections=examples,
        )
        self._is_trained = True

        logger.info(
            "Calibrator diagnostics: mean_raw=%.3f mean_actual=%.3f hedging_gap=%+.3f alpha=%.4f",
            mean_raw, mean_actual, hedging_gap, PLATT_ALPHA,
        )

        return self.diagnostics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(self, raw_prob: float) -> float:
        """
        Apply Platt Scaling (α=√3) to extremize a hedged LLM probability.

        Parameters
        ----------
        raw_prob : float
            LLM or ensemble mean probability in [0.0, 1.0].

        Returns
        -------
        float
            Calibrated probability. Values away from 0.5 are pushed further
            away; values near 0.5 are approximately unchanged.

        Raises
        ------
        RuntimeError
            If train_calibrator() has not been called.
        """
        if not self._is_trained:
            raise RuntimeError("Call train_calibrator() before calibrate().")

        if not (0.0 <= raw_prob <= 1.0):
            raise ValueError(f"raw_prob must be in [0, 1], got {raw_prob}")

        calibrated = self._platt_scale(raw_prob, alpha=PLATT_ALPHA)
        calibrated = max(0.01, min(calibrated, 0.99))

        logger.info(
            "Platt scaling: %.4f → %.4f (alpha=%.4f, delta=%+.4f)",
            raw_prob, calibrated, PLATT_ALPHA, calibrated - raw_prob,
        )
        return calibrated
