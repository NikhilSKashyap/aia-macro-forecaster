"""
src/utils/formatting.py
------------------------
UI helper utilities for formatting and rendering data in the Streamlit app.
"""

from __future__ import annotations


def prob_to_pct(prob: float, decimals: int = 1) -> str:
    """Convert a probability float to a percentage string."""
    return f"{prob * 100:.{decimals}f}%"


def prob_to_label(prob: float) -> str:
    """Return a human-readable confidence label for a probability."""
    if prob >= 0.80:
        return "High"
    elif prob >= 0.60:
        return "Moderate-High"
    elif prob >= 0.45:
        return "Moderate"
    elif prob >= 0.25:
        return "Moderate-Low"
    else:
        return "Low"


def delta_description(raw: float, calibrated: float) -> str:
    """Describe the direction and magnitude of the calibration adjustment."""
    delta = calibrated - raw
    pct_pts = abs(delta) * 100
    direction = "downward" if delta < 0 else "upward"
    return f"{direction} by {pct_pts:.1f} percentage points"
