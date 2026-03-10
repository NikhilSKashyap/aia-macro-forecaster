# AIA Macro Forecaster

> A production-grade prototype implementing and extending the AIA Forecaster architecture
> from *AIA Forecaster: Technical Report* — Alur, Stadie, Kang et al., Bridgewater AIA Labs
> [arXiv:2511.07678](https://arxiv.org/pdf/2511.07678)

---

## Overview

This system is a working implementation of the four-stage judgmental forecasting pipeline introduced in the AIA paper — adapted for macroeconomic binary event forecasting with two ensemble strategies: the paper's original temperature-sampling method, and a cross-model extension that replaces same-model repetition with genuinely diverse model families.

The paper establishes that LLMs can match expert superforecasters at scale by combining agentic search, ensemble averaging, supervisor reconciliation, and statistical calibration. It achieves a Brier score of **0.1076 on ForecastBench (FB-7-21)**, statistically indistinguishable from human superforecasters (p = 0.1522), and **0.0753 on FB-Market** — approaching but not exceeding the market consensus of 0.0965.

This prototype faithfully implements every layer of that pipeline. The ensemble stage is user-configurable across three modes, allowing direct comparison of the two ensemble strategies.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FORECAST PIPELINE                            │
│                                                                     │
│  ┌─────────────┐                                                    │
│  │ MacroData   │  6 themed news snippets + Fed statements           │
│  │ Fetcher     │  (simulates agentic retrieval layer)               │
│  └──────┬──────┘                                                    │
│         │ context injected into all agents simultaneously           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │         STAGE 1 — PARALLEL ENSEMBLE  [configurable]      │       │
│  │                                                          │       │
│  │  Mode A: Cross-Model Diversity (our extension)           │       │
│  │  ┌────────────┐ ┌──────────┐ ┌───────────┐ ┌─────────┐   │       │
│  │  │Claude Haiku│ │ GPT-4o   │ │Grok-4-lat.│ │Gemini   │   │       │
│  │  └────────────┘ └──────────┘ └───────────┘ └─────────┘   │       │
│  │                                                          │       │
│  │  Mode B: Temperature Sampling (paper method, M=3)        │       │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │       │
│  │  │ Claude T=0.3 │ │ Claude T=0.65│ │ Claude T=1.0 │      │       │
│  │  └──────────────┘ └──────────────┘ └──────────────┘      │       │
│  │                                                          │       │
│  │  Mode C: Hybrid (both combined, up to 7 agents)          │       │
│  │                                                          │       │
│  │              Simple Mean  p̄ = (1/n)Σpᵢ                   │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │         STAGE 2 — CLAUDE SONNET SUPERVISOR               │       │
│  │                                                          │       │
│  │  Reads all reasoning traces → identifies disagreements   │       │
│  │  → issues targeted queries → confidence-gates output:    │       │
│  │                                                          │       │
│  │  HIGH confidence  →  supervisor p replaces ensemble mean │       │
│  │  MED / LOW        →  ensemble mean preserved             │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │         STAGE 3 — PLATT SCALING  (α = √3)                │       │
│  │                                                          │       │
│  │         p̂ = σ(√3 · logit(p))                             │       │
│  │                                                          │       │
│  │  Extremizes hedged LLM probabilities away from 0.5       │       │
│  │  Fixed α avoids overfitting on small calibration sets    │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │   STAGE 4 — OPTIONAL MARKET BLEND  (simplex regression)  │       │
│  │                                                          │       │ 
│  │   p_final = α_LLM · p̂_LLM + (1-α_LLM) · p_market         │       │
│  │   default: α_LLM = 0.40  (paper optimal ≈ 0.33 on        │       │
│  │   liquid markets, 0.87 on easier benchmarks)             │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Ensemble Strategies

### Mode B — Temperature Sampling (Paper's Original Method)

The paper runs **M=10 independent calls of the same model** with stochastic sampling to generate forecast diversity. The theoretical justification is Jensen's inequality on Brier score: for a strictly convex loss, the expected loss of the mean is strictly less than the mean of individual losses:

```
ℓ(𝔼[P|Q=q], 𝔼[O|Q=q]) < 𝔼[ℓ(P,O)|Q=q]
```

This implementation uses M=3 (configurable up to 5) with temperatures evenly spaced across [0.3, 1.0]:

| Run | Temperature | Character |
|---|---|---|
| Run 1 | 0.30 | Near-greedy — highest fidelity to dominant evidence |
| Run 2 | 0.65 | Balanced exploration / exploitation |
| Run 3 | 1.00 | High entropy — surfaces minority reasoning paths |

M=3 is cost-efficient for prototyping. The paper's ensemble size analysis (Figure 3) shows diminishing returns past ~10 agents; the marginal gain from M=3 to M=10 is measurable but small relative to the jump from M=1 to M=3.

### Mode A — Cross-Model Diversity (Our Extension)

The Jensen's inequality argument applies to any ensemble, but its practical value depends critically on **forecast independence**. Same-model temperature sampling yields residuals that are correlated through shared RLHF alignment, identical pretraining corpora, and identical inductive biases — particularly on politically or economically sensitive questions where alignment pressures are strongest.

This extension replaces same-model repetition with **four distinct model families**, each independently selectable from the sidebar:

| Agent | Provider | Model | Differentiating characteristic |
|---|---|---|---|
| Agent 1 | Anthropic | `claude-haiku-4-5` | Constitutional AI alignment, strong instruction following |
| Agent 2 | OpenAI | `gpt-4o` | Broad pretraining corpus, RLHF from human contractors |
| Agent 3 | xAI | `grok-4-latest` | Trained on X/Twitter corpus, real-time data orientation |
| Agent 4 | Google | `gemini-2.5-flash` | Multimodal pretraining, distinct document-level reasoning |
| **Supervisor** | **Anthropic** | **`claude-sonnet-4-6`** | **Highest-quality reconciler — reserved for synthesis** |

The theoretical claim: genuine cross-model diversity reduces forecast correlation ρ relative to same-model temperature sampling, yielding larger variance reduction per agent for the same ensemble size.

### Mode C — Hybrid

Both strategies run simultaneously. Up to 7 agents in parallel: 4 cross-model providers + 3 temperature-sampled Claude runs. Maximises ensemble diversity by combining inter-model epistemic diversity with intra-model stochastic variance. The Claude Sonnet supervisor reads all traces regardless of which agents produced them.

---

## The Four Failure Modes (from the Paper)

The paper identifies four systematic failure modes in prior LLM forecasting work. Each is directly addressed:

**1. Non-adaptive search**
Prior work ran fixed keyword queries once. The benefit from search was shown to be negative or negligible without adaptation. The paper's agentic agents condition each query on all prior results, yielding a 7.3% Brier improvement over no search. This implementation simulates this layer via `MacroDataFetcher`; the interface is a drop-in replacement point for any live search client (Bloomberg, FRED, Perplexity) with no other code changes required.

**2. No ensembling**
Single LLM runs have high variance. The paper shows ensembles of 10 agents reduce Brier score substantially. This implementation offers three ensemble configurations: 4-model cross-model diversity, M=3 temperature sampling (paper method), or both combined.

**3. Naive forecast aggregation**
Simply asking an LLM to "average" or "synthesise" other forecasts performs **worse than the simple mean** — models over-weight outlier opinions. The paper's supervisor reframes the task as adversarial evidence review: *identify specific evidentiary disagreements and resolve them*, not synthesise opinions. The Claude Sonnet supervisor uses identical task framing, with confidence-gated output that falls back to the ensemble mean when reconciliation is uncertain.

**4. Uncorrected LLM hedging bias**
RLHF post-training causes LLMs to hedge toward 0.5, expressing uncertainty even when evidence is strong. This is **not** overconfidence — it is underconfidence at the extremes. Platt scaling with fixed α=√3 corrects this by extremizing: pushing 0.65 → 0.745, 0.75 → 0.870, while leaving 0.50 unchanged.

---

## Calibration: Platt Scaling with Fixed α = √3

The calibration transform is closed-form — no fitting required at inference time:

```
p̂ = σ(α · logit(p))

where:
  logit(p) = log(p / (1-p))
  σ(x)     = 1 / (1 + e^(-x))
  α        = √3 ≈ 1.7321  (fixed, not learned)
```

**Correction examples:**

| Raw LLM prob | Calibrated | Δ |
|---|---|---|
| 0.35 | 0.2550 | −9.5 pp |
| 0.45 | 0.4140 | −3.6 pp |
| 0.50 | 0.5000 | 0.0 pp (fixed point) |
| 0.55 | 0.5860 | +3.6 pp |
| 0.65 | 0.7450 | +9.5 pp |
| 0.75 | 0.8702 | +12.0 pp |
| 0.85 | 0.9528 | +10.3 pp |

α is fixed rather than learned. The paper reports the marginal improvement from a learned α is less than 0.005 Brier points (Table 3), while the learned version introduces overfitting risk on small calibration sets. The paper also derives a novel mathematical equivalence (Appendix G.2) between log-odds extremization and Platt scaling on the geometric mean of individual logits — the same transform, two framings.

**Calibration method comparison (Table 3 of the paper):**

| Method | Brier Score |
|---|---|
| No calibration | 0.1140 |
| Platt Scaling (fixed α=√3) | **0.1076** |
| Platt Scaling (learned α) | 0.1071 |
| Log-Odds Extremization (fixed) | 0.1085 |
| Isotonic Regression | 0.1097 |
| OLS | 0.1119 |

---

## Supervisor Agent Design

The supervisor is the most architecturally novel component of the paper. Its correct implementation is non-trivial: asking a model to synthesise opinions degrades performance below the simple mean. The correct framing is adversarial evidence review.

The Claude Sonnet supervisor is explicitly prompted to:

1. Read every agent reasoning trace in full
2. Identify **specific named claims** where agents took different positions
3. Determine which agent's evidence is stronger and why
4. Output a confidence level (`high` / `medium` / `low`)

**Confidence gating (faithful to Section 4.3 of the paper):**
- `high` → supervisor probability replaces ensemble mean
- `medium` / `low` → ensemble mean is preserved; supervisor reasoning logged for transparency

This prevents the supervisor from arbitrarily overriding a well-functioning ensemble when it has no genuine evidentiary grounds to do so. In the hybrid mode, the supervisor reads traces from both cross-model and temperature-sampled agents simultaneously.

---

## Key Results from the Paper

| Forecaster | FB-Market | FB-7-21 | FB-8-14 | MarketLiquid |
|---|---|---|---|---|
| Market Consensus | 0.0965 | — | — | 0.1106 |
| Human Superforecasters | 0.0740 | 0.1110 | 0.1152 | — |
| Prior SOTA LLM | 0.107 | 0.133 | 0.145 | — |
| OpenAI o3 | 0.1096 | 0.1221 | 0.1262 | 0.1324 |
| **AIA Forecaster (paper)** | **0.0753** | **0.1076** | **0.1099** | **0.1258** |

The AIA system is statistically indistinguishable from superforecasters on FB-7-21 (p = 0.1522) and FB-8-14 (p ≥ 0.0946). On FB-Market it approaches but does not statistically exceed market consensus. On the harder MarketLiquid benchmark, the optimal LLM-to-market weight is 33% LLM / 67% market — the LLM still provides diversifying signal even when underperforming the market outright.

**Critical finding on search:** Without agentic search, the system performs at Brier = 0.3609 on live closed-market questions — worse than the 0.25 baseline of guessing 0.5 on every question. Search is not optional.

---

## Project Structure

```
aia-macro-forecaster/
│
├── app.py                          # Streamlit dashboard — pipeline orchestration + UI
│                                   # Ensemble strategy picker, per-model toggles,
│                                   # per-model temperature sliders, market blend,
│                                   # Plotly ensemble chart
│
├── src/
│   ├── data/
│   │   └── client.py               # MacroDataFetcher — themed snippet retrieval
│   │                               # Drop-in replacement point for live search APIs
│   │
│   ├── models/
│   │   ├── providers.py            # Multi-provider abstraction layer
│   │   │                           # _extract_probability_from_text() — regex fallback
│   │   │                           # BaseForecaster (ABC)
│   │   │                           # ClaudeForecaster      — Anthropic tool-use
│   │   │                           # OpenAIForecaster      — GPT-4o JSON mode
│   │   │                           # XAIForecaster         — Grok via OpenAI-compat API
│   │   │                           # GeminiForecaster      — google-generativeai
│   │   │                           # TemperatureSampledForecaster — paper's M=3 method
│   │   │                           # build_cross_model_agents(temperatures={...})
│   │   │                           # build_temperature_agents(temperatures=[...])
│   │   │                           # build_ensemble_agents()  — unified factory
│   │   │
│   │   ├── ensemble.py             # ThreadPoolExecutor parallel orchestrator
│   │   │                           # EnsembleResult: mean, std, median, per-agent output
│   │   │
│   │   ├── supervisor.py           # Claude Sonnet supervisor agent
│   │   │                           # Confidence-gated reconciliation (high/medium/low)
│   │   │                           # Works across all ensemble modes
│   │   │
│   │   └── calibration.py         # Platt Scaling — fixed α=√3
│   │                               # Closed-form extremization, calibration diagnostics
│   │
│   └── utils/
│       └── formatting.py           # Probability formatting helpers
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

**1. Clone and create environment**
```bash
git clone <repo-url>
cd aia-macro-forecaster
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure API keys**
```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...   # Required: Claude Haiku agent + Claude Sonnet supervisor
                               # Also used for temperature-sampling runs
OPENAI_API_KEY=sk-proj-...     # GPT-4o cross-model agent
XAI_API_KEY=xai-...            # Grok-4-latest cross-model agent
GEMINI_API_KEY=AIza...         # Gemini 2.5 Flash cross-model agent
```

The system degrades gracefully if any key is missing — it runs the ensemble over whichever agents are available. Only `ANTHROPIC_API_KEY` is required (needed for both the Claude agent and the supervisor). A minimum of two successful agents is enforced before the pipeline proceeds.

**3. Run**
```bash
streamlit run app.py
```

---

## UI Configuration

The sidebar exposes full control over every aspect of the ensemble:

**Ensemble Strategy**
- *Cross-Model Diversity* — per-provider checkboxes to include/exclude any of the four model families; each enabled provider has an independent temperature slider (0.0–1.0, step 0.05) shown inline
- *Temperature Sampling (paper M=3)* — Claude model selector (Sonnet / Haiku), M slider (2–5), and an individual temperature slider per run (pre-populated with evenly-spaced defaults, fully adjustable)
- *Hybrid* — both strategies simultaneously; all controls available

**Temperature controls — valid ranges by provider:**

| Provider | API temperature range |
|---|---|
| Anthropic Claude | 0.0 – 1.0 |
| OpenAI GPT-4o | 0.0 – 2.0 |
| xAI Grok | 0.0 – 2.0 |
| Google Gemini | 0.0 – 2.0 |

The UI sliders cap at 1.0 for consistency; each provider's constructor clamps to its own API limit internally.

**Market Prior** — optional prediction market price input; blended with calibrated LLM output at 40% LLM / 60% market (paper optimal ≈ 33/67 on liquid markets)

**Quick Examples** — one-click event prefills for common macro scenarios

---

## Dependencies

```
anthropic>=0.40.0           # Claude Haiku agent + Sonnet supervisor + tool-use output
openai>=1.50.0              # GPT-4o agent; also used for xAI Grok (OpenAI-compat endpoint)
google-generativeai>=0.8.0  # Gemini 2.5 Flash agent
streamlit>=1.40.0           # Dashboard
pydantic>=2.9.0             # Structured output validation
scikit-learn>=1.5.0         # Calibration diagnostics
plotly>=5.24.0              # Ensemble visualisation (per-agent bar chart)
pandas>=2.2.0
numpy>=1.26.0
python-dotenv>=1.0.0
```

---

## Robustness & Error Handling

Provider API calls are inherently unreliable — models occasionally return malformed output, drop required fields, or hit rate limits. The system handles this at three levels:

**Agent-level recovery (per API call)**

When a model returns a tool response missing `raw_probability`, the system attempts three recovery strategies in order before marking the agent as failed:

1. **Alternate key names** — checks `probability` and `p` as fallback keys
2. **Regex extraction from `reasoning_chain`** — three patterns in priority order:
   - Decimal after probability keyword: `"probability of 0.72"`, `"estimate: 0.65"`
   - Percentage phrase: `"65%"`, `"roughly 70 percent"`
   - Bare decimal in range [0.05, 0.95] anywhere in the text
3. **Descriptive error** — if all three fail, logs keys present and marks agent failed

**Ensemble-level resilience**

Failed agents are excluded from the mean calculation. The pipeline continues as long as ≥ 2 agents succeed. Failed agents are listed in the UI under a collapsible warning panel.

**Temperature clamping**

Each provider's constructor clamps temperature to its API's valid range (0–1 for Claude, 0–2 for others) regardless of slider input, preventing `400 invalid_request_error` failures.

---

## Known Gaps vs. Production AIA System

| Component | Paper | This prototype | Impact |
|---|---|---|---|
| Live agentic search | Two commercial search APIs, iterative adaptive queries | Simulated themed snippet library | **High** — search is the dominant driver (Brier 0.36 without it on live questions) |
| Ensemble size | M=10 per question | M=3–7 depending on mode | Low-medium — diminishing returns past ~10; cross-model diversity compensates |
| Foreknowledge detection | LLM-as-judge pipeline, 1.65% contamination rate | Not implemented | Medium for live search; irrelevant for simulated data |
| Evaluation framework | ForecastBench + MarketLiquid, Brier scoring | Not implemented | — |

The search layer is the most critical gap. `MacroDataFetcher.fetch_recent_context()` in `src/data/client.py` is the sole replacement point — swapping the snippet simulation for a live search client requires changing only that method body, with no other structural changes.

---

## Reference

```bibtex
@techreport{alur2025aia,
  title     = {AIA Forecaster: Technical Report},
  author    = {Alur, Rohan and Stadie, Bradly C. and Kang, Daniel and Chen, Ryan
               and McManus, Matt and Rickert, Michael and Lee, Tyler and Federici, Michael
               and Zhu, Richard and Fogerty, Dennis and Williamson, Hayley and Lozinski, Nina
               and Linsky, Aaron and Sekhon, Jasjeet S.},
  institution = {Bridgewater AIA Labs},
  year      = {2025},
  url       = {https://arxiv.org/abs/2511.07678}
}
```

---

*Built as a research prototype. Not investment advice. All forecasts are probabilistic estimates and should be interpreted accordingly.*
