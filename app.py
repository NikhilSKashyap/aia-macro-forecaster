"""
app.py
------
AIA Macro Forecaster — Streamlit Dashboard

Full four-stage pipeline per the AIA paper (arXiv 2511.07678):
  Stage 1 — Parallel multi-provider ensemble (Claude, GPT-4o, Grok, Gemini)
  Stage 2 — Claude Sonnet supervisor reconciliation (confidence-gated)
  Stage 3 — Platt Scaling calibration (fixed α=√3, extremizes LLM hedging)
  Stage 4 — Optional market price simplex blend

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from src.data.client import MacroDataFetcher
from src.models.calibration import ProbabilityCalibrator
from src.models.ensemble import EnsembleOrchestrator, EnsembleResult
from src.models.providers import build_ensemble_agents, _PROVIDER_REGISTRY
from src.models.supervisor import SupervisorAgent, SupervisorResult
from src.utils.formatting import delta_description, prob_to_label, prob_to_pct

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("aia_macro_forecaster")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AIA Macro Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.main .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

.aia-header {
    background: linear-gradient(135deg, #0a1628 0%, #0f2d4a 60%, #1a4a6e 100%);
    border-radius: 14px;
    padding: 1.8rem 2.4rem;
    margin-bottom: 1.6rem;
    border-left: 5px solid #3a8fd4;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.aia-header h1 { color: #e8f4fd; margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px; }
.aia-header p  { color: #7ab3d4; margin: 0.4rem 0 0; font-size: 0.88rem; }

.agent-card {
    background: #ffffff;
    border: 1px solid #dce6f0;
    border-top: 3px solid var(--accent, #3a8fd4);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    position: relative;
}
.agent-card .provider  { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--accent, #3a8fd4); }
.agent-card .model-id  { font-size: 0.75rem; color: #888; margin-bottom: 0.5rem; }
.agent-card .prob-big  { font-size: 2rem; font-weight: 800; color: #1a2e40; line-height: 1; }
.agent-card .latency   { font-size: 0.72rem; color: #aaa; margin-top: 0.3rem; }

.supervisor-box {
    background: linear-gradient(135deg, #0f2d4a 0%, #0a1628 100%);
    border: 1px solid #1e4a6e;
    border-left: 4px solid #f0b429;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    color: #c9dce8;
    margin-bottom: 1rem;
}
.supervisor-box h4 { color: #f0b429; margin: 0 0 0.6rem; font-size: 0.95rem; }

.calibration-note {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.84rem;
    color: #5d4037;
}

.snippet-card {
    background: #ffffff;
    border: 1px solid #d1dce8;
    border-left: 4px solid #3a8fd4;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.87rem;
    line-height: 1.55;
}
.snippet-source   { color: #3a8fd4; font-weight: 700; font-size: 0.75rem; text-transform: uppercase; }
.snippet-headline { color: #1a2e40; font-weight: 700; margin: 0.2rem 0; font-size: 0.9rem; }
.snippet-body     { color: #4a5a6a; }

.pipeline-step {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #e8f2fd; border: 1px solid #b0d0f0;
    border-radius: 20px; padding: 0.25rem 0.8rem;
    font-size: 0.78rem; font-weight: 600; color: #1a3a5c;
    margin-right: 0.4rem; margin-bottom: 0.4rem;
}
.pipeline-arrow { color: #aaa; font-size: 1rem; margin: 0 0.1rem; }

[data-testid="stMetricLabel"]  { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"]  { font-size: 2.2rem !important; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Provider accent colours
# ---------------------------------------------------------------------------
PROVIDER_COLORS = {
    "Anthropic Claude": "#cc785c",
    "OpenAI GPT-4o":   "#10a37f",
    "xAI Grok":        "#1da1f2",
    "Google Gemini":   "#ea4335",
    # Temperature-sampled runs get a blue gradient by run index
    "Claude Run 1":    "#3a5fc8",
    "Claude Run 2":    "#5b7ee0",
    "Claude Run 3":    "#7c9ef8",
    "Claude Run 4":    "#9db8ff",
    "Claude Run 5":    "#becfff",
}

def _agent_color(provider_name: str) -> str:
    """Return accent colour for a provider, with fallback for temperature agents."""
    for key, color in PROVIDER_COLORS.items():
        if provider_name.startswith(key):
            return color
    return "#888888"

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_calibrator() -> ProbabilityCalibrator:
    cal = ProbabilityCalibrator()
    cal.train_calibrator(n_samples=200, random_seed=42)
    return cal

def get_ensemble_orchestrator(
    strategy: str,
    enabled_providers: list,
    provider_temperatures: dict,
    temp_model: str,
    m: int,
    sampling_temperatures: list,
) -> EnsembleOrchestrator:
    """Build a fresh orchestrator based on user-selected ensemble config."""
    agents = build_ensemble_agents(
        strategy=strategy,
        enabled_providers=enabled_providers if strategy != "temperature" else None,
        provider_temperatures=provider_temperatures,
        temp_model=temp_model,
        m=m,
        sampling_temperatures=sampling_temperatures if sampling_temperatures else None,
    )
    return EnsembleOrchestrator(agents=agents, max_workers=len(agents))

@st.cache_resource(show_spinner=False)
def get_supervisor() -> Optional[SupervisorAgent]:
    try:
        return SupervisorAgent()
    except Exception as exc:
        logger.warning("Supervisor unavailable: %s", exc)
        return None

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "ensemble_result": None,
        "supervisor_result": None,
        "calibrated_prob": None,
        "snippets": None,
        "last_event": "",
        "pipeline_error": None,
        "market_blended": None,
        "ensemble_strategy": "cross_model",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    event: str,
    market_price: Optional[float],
    ensemble_strategy: str,
    enabled_providers: list,
    provider_temperatures: dict,
    temp_model: str,
    m: int,
    sampling_temperatures: list,
) -> None:
    """Execute all four pipeline stages and store results in session state."""
    for k in ["ensemble_result","supervisor_result","calibrated_prob",
              "snippets","pipeline_error","market_blended"]:
        st.session_state[k] = None

    fetcher    = MacroDataFetcher(num_snippets=6)
    supervisor = get_supervisor()
    calibrator = get_calibrator()

    try:
        orchestrator = get_ensemble_orchestrator(
            strategy=ensemble_strategy,
            enabled_providers=enabled_providers,
            provider_temperatures=provider_temperatures,
            temp_model=temp_model,
            m=m,
            sampling_temperatures=sampling_temperatures,
        )
    except Exception as exc:
        st.session_state.pipeline_error = f"Ensemble configuration failed: {exc}"
        return

    status_box = st.empty()

    # ── Stage 1: Data ingestion ──────────────────────────────────────────
    with status_box.container():
        st.info("📡  **Stage 1/4** — Fetching macroeconomic context snippets…")
    try:
        snippets = fetcher.fetch_recent_context(event)
        st.session_state.snippets = snippets
    except Exception as exc:
        st.session_state.pipeline_error = f"Data fetch failed: {exc}"
        status_box.empty()
        return

    # ── Stage 2: Parallel ensemble ───────────────────────────────────────
    strategy_label = {
        "cross_model": "cross-model diversity",
        "temperature": f"temperature sampling (M={m})",
        "hybrid":      f"hybrid (cross-model + M={m} temperature)",
    }.get(ensemble_strategy, ensemble_strategy)

    with status_box.container():
        st.info(
            f"🤖  **Stage 2/4** — Running parallel ensemble "
            f"({len(orchestrator.agents)} agents · {strategy_label})…  "
            "_All agents run simultaneously._"
        )
    try:
        ensemble: EnsembleResult = orchestrator.run(event, snippets)
        st.session_state.ensemble_result = ensemble
    except Exception as exc:
        err = str(exc)
        if "credit balance" in err.lower():
            st.session_state.pipeline_error = "BILLING: Add credits at console.anthropic.com/billing"
        else:
            st.session_state.pipeline_error = f"Ensemble failed: {exc}"
        status_box.empty()
        return

    # ── Stage 3: Supervisor reconciliation ──────────────────────────────
    with status_box.container():
        st.info("🧠  **Stage 3/4** — Claude Sonnet supervisor reconciling agent disagreements…")
    sup_result: Optional[SupervisorResult] = None
    if supervisor:
        try:
            sup_result = supervisor.reconcile(event, ensemble)
            st.session_state.supervisor_result = sup_result
        except Exception as exc:
            logger.warning("Supervisor failed, using ensemble mean: %s", exc)
            # Non-fatal: fall through to ensemble mean
    else:
        logger.warning("No supervisor available; using ensemble mean.")

    pre_calibration_prob = (
        sup_result.final_probability if sup_result else ensemble.mean_probability
    )

    # ── Stage 4: Calibration ─────────────────────────────────────────────
    with status_box.container():
        st.info("📐  **Stage 4/4** — Applying Platt Scaling (α=√3)…")
    time.sleep(0.2)
    try:
        calibrated = calibrator.calibrate(pre_calibration_prob)
        st.session_state.calibrated_prob = calibrated
    except Exception as exc:
        st.session_state.pipeline_error = f"Calibration failed: {exc}"
        status_box.empty()
        return

    # ── Optional: market blend ───────────────────────────────────────────
    if market_price is not None:
        # Simplex blend: equal weight by default (paper uses 0.33 LLM / 0.67 market
        # for liquid markets; we expose α=0.5 as a reasonable prior)
        alpha_llm = 0.40
        blended = alpha_llm * calibrated + (1 - alpha_llm) * market_price
        st.session_state.market_blended = round(blended, 4)

    st.session_state.last_event = event
    status_box.empty()

# ---------------------------------------------------------------------------
# Plotly ensemble chart
# ---------------------------------------------------------------------------

def _ensemble_chart(ensemble: EnsembleResult, sup_result: Optional[SupervisorResult]) -> go.Figure:
    """
    Horizontal bar chart showing each agent's probability alongside
    ensemble mean and (if applicable) supervisor reconciliation.
    """
    agents   = ensemble.successful_agents
    names    = [f"{a.provider}" for a in agents]
    probs    = [a.raw_probability for a in agents]
    colors   = [_agent_color(a.provider) for a in agents]
    latency  = [f"{a.latency_ms:.0f} ms" for a in agents]

    fig = go.Figure()

    # Agent bars
    fig.add_trace(go.Bar(
        y=names,
        x=probs,
        orientation="h",
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0.15)",
        marker_line_width=1,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        customdata=latency,
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.3f}<br>Latency: %{customdata}<extra></extra>",
        name="Agent Forecast",
    ))

    # Ensemble mean line
    fig.add_vline(
        x=ensemble.mean_probability,
        line_dash="dash",
        line_color="#1a3a5c",
        line_width=2,
        annotation_text=f"Ensemble mean: {ensemble.mean_probability:.3f}",
        annotation_position="top right",
        annotation_font_color="#1a3a5c",
        annotation_font_size=11,
    )

    # Supervisor line (if high-confidence override)
    if sup_result and sup_result.used_supervisor_output:
        fig.add_vline(
            x=sup_result.reconciled_probability,
            line_dash="dot",
            line_color="#f0b429",
            line_width=2,
            annotation_text=f"Supervisor: {sup_result.reconciled_probability:.3f}",
            annotation_position="bottom right",
            annotation_font_color="#c88b00",
            annotation_font_size=11,
        )

    fig.update_layout(
        height=180 + len(agents) * 42,
        margin=dict(l=10, r=60, t=30, b=20),
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#e8ecf0",
            title="Probability",
        ),
        yaxis=dict(showgrid=False),
        plot_bgcolor="#fafbfc",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family="sans-serif", size=12),
    )
    return fig

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_agent_cards(ensemble: EnsembleResult) -> None:
    # Cap at 6 columns — beyond that cards become too narrow
    agents = ensemble.successful_agents
    n_cols = min(len(agents), 6)
    cols = st.columns(n_cols)
    for i, agent in enumerate(agents):
        color = _agent_color(agent.provider)
        with cols[i % n_cols]:
            st.markdown(
                f"""<div class="agent-card" style="--accent:{color}">
                  <div class="provider">{agent.provider}</div>
                  <div class="model-id">{agent.model}</div>
                  <div class="prob-big">{agent.raw_probability:.1%}</div>
                  <div class="latency">⏱ {agent.latency_ms:.0f} ms</div>
                </div>""",
                unsafe_allow_html=True,
            )

def _render_supervisor(sup: SupervisorResult) -> None:
    confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(sup.confidence, "⚪")
    override_note = (
        f"**Override applied** — supervisor's {sup.reconciled_probability:.1%} "
        f"replaced ensemble mean of {sup.ensemble_mean:.1%}"
        if sup.used_supervisor_output
        else f"**Deferred to ensemble mean** — supervisor confidence is {sup.confidence}"
    )
    st.markdown(
        f"""<div class="supervisor-box">
          <h4>🧠 Claude Sonnet Supervisor &nbsp; {confidence_emoji} {sup.confidence.upper()} confidence</h4>
          <p style="font-size:0.82rem; margin:0 0 0.6rem">{override_note}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    if sup.disagreements_identified:
        with st.expander("Disagreements identified by supervisor", expanded=False):
            for d in sup.disagreements_identified:
                st.markdown(f"- {d}")

    with st.expander("Supervisor reconciliation reasoning", expanded=False):
        st.markdown(sup.reconciliation_reasoning)

    if sup.key_evidence_gaps:
        with st.expander("Evidence gaps (unresolved)", expanded=False):
            for g in sup.key_evidence_gaps:
                st.markdown(f"- {g}")

def _render_final_metrics(
    ensemble: EnsembleResult,
    sup: Optional[SupervisorResult],
    calibrated: float,
    market_blended: Optional[float],
    market_price: Optional[float],
) -> None:
    pre_cal = sup.final_probability if sup else ensemble.mean_probability
    delta_pp = (calibrated - pre_cal) * 100

    cols = st.columns(4 if market_blended else 3)

    with cols[0]:
        st.metric(
            "Ensemble Mean",
            prob_to_pct(ensemble.mean_probability),
            help="Simple mean of all successful agent forecasts (paper: Jensen's inequality guarantees this < mean individual loss under Brier score)."
        )
        st.caption(f"σ = {ensemble.std_probability:.3f} · {ensemble.agent_disagreement_level} disagreement")

    with cols[1]:
        sup_label = sup.reconciled_probability if sup and sup.used_supervisor_output else pre_cal
        used = sup.used_supervisor_output if sup else False
        st.metric(
            "Supervisor Output",
            prob_to_pct(sup_label),
            delta=f"{'overrode' if used else 'deferred'} mean",
            delta_color="off",
            help="Claude Sonnet's reconciled estimate. Replaces ensemble mean only on high-confidence reconciliations."
        )
        st.caption(f"Confidence: **{sup.confidence.upper()}**" if sup else "Not available")

    with cols[2]:
        st.metric(
            "Calibrated (Platt α=√3)",
            prob_to_pct(calibrated),
            delta=f"{delta_pp:+.1f} pp vs pre-cal",
            delta_color="inverse" if delta_pp < 0 else "normal",
            help="Platt Scaling extremizes hedged LLM probabilities. α=√3 is the fixed parameter from the AIA paper."
        )
        st.caption(f"Label: **{prob_to_label(calibrated)}**")

    if market_blended and market_price is not None:
        with cols[3]:
            st.metric(
                "Market Blend (60/40)",
                prob_to_pct(market_blended),
                help="Simplex blend: 40% calibrated LLM + 60% market consensus. Paper found optimal LLM weight ≈ 33% on liquid markets."
            )
            st.caption(f"Market prior: {prob_to_pct(market_price)}")

def _render_agent_reasoning(ensemble: EnsembleResult) -> None:
    st.markdown("#### Per-Agent Reasoning Chains")
    for agent in ensemble.successful_agents:
        color = PROVIDER_COLORS.get(agent.provider, "#888")
        with st.expander(f"{agent.provider} ({agent.model}) — {agent.raw_probability:.1%}", expanded=False):
            if agent.key_factors:
                st.markdown("**Key factors:**")
                for f in agent.key_factors:
                    st.markdown(f"- {f}")
            st.markdown("**Reasoning:**")
            st.markdown(agent.reasoning_chain)

def _render_snippets(snippets) -> None:
    st.markdown("#### Context Window (6 snippets injected into all agents)")
    for s in snippets:
        st.markdown(
            f"""<div class="snippet-card">
              <div class="snippet-source">{s.source} · {s.date}</div>
              <div class="snippet-headline">{s.headline}</div>
              <div class="snippet-body">{s.body}</div>
            </div>""",
            unsafe_allow_html=True,
        )

def _render_calibration_sidebar(cal: ProbabilityCalibrator) -> None:
    diag = cal.diagnostics
    if not diag:
        return
    with st.sidebar.expander("📐 Calibration Model", expanded=False):
        st.markdown(f"**Method:** Platt Scaling (fixed α = {diag.platt_alpha:.4f} = √3)")
        st.markdown(f"**Direction:** Extremization (corrects LLM hedging toward 0.5)")
        st.markdown("---")
        st.markdown("**Example corrections:**")
        for raw, cal_p in diag.example_corrections:
            direction = "↑" if cal_p > raw else "↓"
            st.markdown(f"- {raw:.2f} → **{cal_p:.4f}** {direction}")
        st.caption(
            "Values away from 0.5 are pushed further away. "
            "α=√3 is fixed (not learned) per Section 4.4 of the AIA paper "
            "to avoid overfitting on small calibration sets."
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _init_state()

    # Header
    st.markdown("""
    <div class="aia-header">
      <h1>📈 AIA Macro Forecaster</h1>
      <p>
        Multi-provider LLM ensemble · Claude Sonnet supervisor · Platt Scaling calibration<br>
        Based on <em>AIA Forecaster: Technical Report</em> (Alur et al., 2025 · arXiv:2511.07678) · Bridgewater AIA Labs
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Forecast Configuration")
        st.markdown("---")

        event_input = st.text_area(
            "Target Macroeconomic Event",
            value="Will the Federal Reserve cut interest rates by 25bps at the next FOMC meeting?",
            height=90,
            help="Binary yes/no macro event. Be specific — the more precise the question, the better the forecast.",
        )

        # ── Ensemble strategy ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Ensemble Strategy")

        strategy_labels = {
            "Cross-Model Diversity (our extension)": "cross_model",
            "Temperature Sampling (paper M=3)":      "temperature",
            "Hybrid (both combined)":                "hybrid",
        }
        strategy_choice = st.radio(
            "Method",
            list(strategy_labels.keys()),
            index=0,
            help=(
                "**Cross-Model:** 4 distinct LLM families run in parallel — "
                "lower residual correlation than same-model repetition.\n\n"
                "**Temperature Sampling:** paper's original method — one Claude model "
                "run M times at different temperatures (T=0.3, 0.7, 1.1).\n\n"
                "**Hybrid:** both strategies combined for maximum ensemble diversity."
            ),
        )
        ensemble_strategy = strategy_labels[strategy_choice]

        # Cross-model provider toggles + per-provider temperature sliders
        enabled_providers: list = list(_PROVIDER_REGISTRY.keys())
        provider_temperatures: dict = {}
        if ensemble_strategy in ("cross_model", "hybrid"):
            st.markdown("**Providers & temperatures:**")
            enabled_providers = []
            provider_list = list(_PROVIDER_REGISTRY.keys())
            provider_emojis = {
                "Claude Haiku": "🟠", "OpenAI GPT-4o": "🟢",
                "xAI Grok-4": "🔵", "Google Gemini": "🔴",
            }
            for provider in provider_list:
                emoji = provider_emojis.get(provider, "⚪")
                col_cb, col_sl = st.columns([0.38, 0.62])
                with col_cb:
                    enabled = st.checkbox(
                        f"{emoji} {provider}", value=True, key=f"cb_{provider}",
                    )
                with col_sl:
                    t = st.slider(
                        "T", min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                        key=f"temp_{provider}",
                        disabled=not enabled,
                        label_visibility="collapsed",
                        help=f"Sampling temperature for {provider}. Range 0–1 (Claude/Gemini) or 0–2 (OpenAI/Grok).",
                    )
                if enabled:
                    enabled_providers.append(provider)
                    provider_temperatures[provider] = t
            if not enabled_providers:
                st.warning("Select at least one provider.")

        # Temperature sampling controls + per-run sliders
        temp_model = "claude-sonnet-4-6"
        m_value = 3
        sampling_temperatures: list = []
        if ensemble_strategy in ("temperature", "hybrid"):
            st.markdown("**Temperature sampling:**")
            temp_model = st.selectbox(
                "Claude model",
                ["claude-sonnet-4-6", "claude-haiku-4-5"],
                index=0,
                help="claude-sonnet-4-6 gives higher quality; claude-haiku-4-5 is faster and cheaper.",
            )
            m_value = st.slider(
                "M (samples)", min_value=2, max_value=5, value=3, step=1,
                help="Paper uses M=10. M=3 is cost-efficient. Each run gets its own temperature below.",
            )
            # Default even spacing, user can adjust each run individually
            if m_value > 1:
                default_step = (1.0 - 0.3) / (m_value - 1)
                default_temps = [round(0.3 + default_step * i, 2) for i in range(m_value)]
            else:
                default_temps = [0.7]

            st.markdown("**Per-run temperatures:**")
            for i, default_t in enumerate(default_temps, start=1):
                t = st.slider(
                    f"Run {i}",
                    min_value=0.0, max_value=1.0,
                    value=default_t, step=0.05,
                    key=f"sampling_temp_{i}",
                    help=f"Temperature for Claude sampling run {i}.",
                )
                sampling_temperatures.append(t)

        # ── Market prior ─────────────────────────────────────────────────
        st.markdown("---")
        use_market = st.checkbox("Include market prior", value=False)
        market_price_input: Optional[float] = None
        if use_market:
            market_price_input = st.slider(
                "Prediction market price",
                min_value=0.01, max_value=0.99, value=0.50, step=0.01,
                help="Current market-implied probability from Polymarket / Kalshi. Paper: optimal LLM weight ≈ 33% on liquid markets.",
            )

        # ── Quick examples ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Quick examples:**")
        examples = [
            "Will the US enter recession in 2025?",
            "Will headline CPI fall below 2.5% by year-end?",
            "Will 10-year Treasury yields exceed 5% this year?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:15]}", use_container_width=True):
                st.session_state["_prefill"] = ex
                st.rerun()
        if "_prefill" in st.session_state:
            event_input = st.session_state.pop("_prefill")

        st.markdown("---")

        # Disable run if cross-model selected but no providers ticked
        can_run = bool(event_input.strip()) and (
            ensemble_strategy == "temperature"
            or bool(enabled_providers)
        )
        run_btn = st.button(
            "🚀  Run Forecast Pipeline",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

        st.markdown("---")
        _render_calibration_sidebar(get_calibrator())

        st.caption(
            "**Supervisor:** Claude Sonnet 4.6  \n"
            "**Calibration:** Platt Scaling α=√3"
        )

    # Trigger pipeline
    if run_btn and can_run:
        run_pipeline(
            event=event_input.strip(),
            market_price=market_price_input,
            ensemble_strategy=ensemble_strategy,
            enabled_providers=enabled_providers,
            provider_temperatures=provider_temperatures,
            temp_model=temp_model,
            m=m_value,
            sampling_temperatures=sampling_temperatures,
        )

    # Error display
    if st.session_state.pipeline_error:
        st.error(f"**Pipeline error:** {st.session_state.pipeline_error}")
        if "BILLING" in st.session_state.pipeline_error:
            st.info("💳 Add credits at **console.anthropic.com/billing** then retry.")
        st.stop()

    # Results
    ensemble: Optional[EnsembleResult] = st.session_state.ensemble_result
    sup: Optional[SupervisorResult]    = st.session_state.supervisor_result
    calibrated: Optional[float]        = st.session_state.calibrated_prob
    snippets                           = st.session_state.snippets
    market_blended: Optional[float]    = st.session_state.market_blended

    if ensemble is None or calibrated is None:
        # Landing state
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color: #7a9ab8;">
          <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
          <p style="font-size:1.1rem;">
            Configure your event in the sidebar and click
            <strong>Run Forecast Pipeline</strong>.
          </p>
          <div style="margin-top:1.5rem; display:flex; justify-content:center; gap:0.3rem; flex-wrap:wrap;">
            <span class="pipeline-step">📡 Data ingestion</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">🤖 Parallel ensemble</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">🧠 Claude supervisor</span>
            <span class="pipeline-arrow">→</span>
            <span class="pipeline-step">📐 Platt scaling</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Results ─────────────────────────────────────────────────────────
    st.markdown(f"### Forecast: *{st.session_state.last_event}*")
    st.markdown("---")

    # Agent probability cards
    st.markdown("#### Agent Ensemble")
    _render_agent_cards(ensemble)

    # Plotly chart
    st.plotly_chart(
        _ensemble_chart(ensemble, sup),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown("---")

    # Supervisor section
    if sup:
        _render_supervisor(sup)
    else:
        st.info("Supervisor unavailable — using ensemble mean as pre-calibration input.")

    st.markdown("---")

    # Final probability metrics
    st.markdown("#### Final Probability Estimates")
    _render_final_metrics(
        ensemble=ensemble,
        sup=sup,
        calibrated=calibrated,
        market_blended=market_blended,
        market_price=market_price_input,
    )

    st.markdown("---")

    # Collapsible sections
    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        _render_agent_reasoning(ensemble)
    with col_right:
        if snippets:
            _render_snippets(snippets)

    # Failed agents (if any)
    if ensemble.failed_agents:
        with st.expander(f"⚠️ {ensemble.n_failed} agent(s) failed", expanded=False):
            for a in ensemble.failed_agents:
                st.markdown(f"- **{a.provider}**: `{a.error}`")


if __name__ == "__main__":
    main()
