"""
src/models/providers.py
------------------------
Multi-provider forecaster abstraction layer.

Each provider (Anthropic Claude, OpenAI GPT-4o, xAI Grok, Google Gemini) is
wrapped in a concrete implementation of BaseForecaster. All return a common
AgentForecast dataclass so the EnsembleOrchestrator can treat them uniformly.

Design rationale
----------------
The paper runs M=10 independent agents and relies on variance reduction through
ensembling (Jensen's inequality on Brier score). We implement a stronger version:
instead of running the same model 10 times, we use 4 DISTINCT model families.
This produces genuinely uncorrelated forecast errors — different training data,
different RLHF alignment, different world models — yielding greater ensemble
diversity than same-model repetition.

    Agent 1: claude-haiku-4-5      (Anthropic)   — speed-optimised agent
    Agent 2: gpt-4o                (OpenAI)       — broad world knowledge
    Agent 3: grok-4-latest         (xAI)          — real-time data focus
    Agent 4: gemini-2.5-flash      (Google)       — long-context synthesis
    Supervisor: claude-sonnet-4-6  (Anthropic)    — highest-quality reconciler
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic
import google.generativeai as genai
from openai import OpenAI

from src.data.client import NewsSnippet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class AgentForecast:
    """
    Output of a single provider agent.

    Fields
    ------
    provider      : Human-readable provider name (e.g. "OpenAI GPT-4o")
    model         : Exact model ID used
    reasoning_chain : Chain-of-thought text
    key_factors   : List of 3-6 driving factors
    raw_probability : Probability in [0, 1] before calibration
    latency_ms    : Wall-clock time for the API call
    error         : Error message if the call failed; None on success
    """
    provider: str
    model: str
    reasoning_chain: str
    key_factors: List[str]
    raw_probability: float
    latency_ms: float
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Shared prompt content
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are AIA-Macro, an objective macroeconomic probability forecaster.
Your task is to read the provided news snippets and estimate the probability
that the stated event will occur.

## Calibration Discipline
- You are known to hedge predictions toward 0.5. Resist this bias.
- If evidence strongly supports the event, assign a probability above 0.65.
- If evidence strongly opposes the event, assign a probability below 0.35.
- Only values in the 0.40–0.60 range are appropriate when evidence is
  genuinely mixed or absent.

## Output
Return a single valid JSON object with exactly these keys:
{
  "reasoning_chain": "<your step-by-step analysis>",
  "key_factors": ["<factor 1>", "<factor 2>", ..., "<factor N>"],
  "raw_probability": <float between 0.0 and 1.0>
}
Return ONLY the JSON object — no preamble, no markdown fences.
"""

def _build_user_message(event: str, snippets: List[NewsSnippet]) -> str:
    snippet_block = "\n\n---\n\n".join(str(s) for s in snippets)
    return (
        f"## Event to Forecast\n\n{event}\n\n"
        f"## Context Snippets\n\n{snippet_block}\n\n"
        f"Now return your JSON forecast."
    )

def _parse_json_response(text: str, provider: str) -> dict:
    """
    Extract JSON from model output, stripping markdown fences if present.
    Raises ValueError if parsing fails.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"[{provider}] No JSON object found in response: {text[:200]}")
    return json.loads(match.group())


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseForecaster(ABC):
    """Abstract interface for all provider agents."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def _call_api(self, event: str, snippets: List[NewsSnippet]) -> dict:
        """
        Make the provider-specific API call.
        Must return a dict with keys: reasoning_chain, key_factors, raw_probability.
        Raises on API error.
        """
        ...

    def forecast(self, event: str, snippets: List[NewsSnippet]) -> AgentForecast:
        """
        Public method: call the API, parse output, return AgentForecast.
        Catches all errors and returns a failed AgentForecast rather than raising,
        so the ensemble can continue even if one provider is unavailable.
        """
        t0 = time.perf_counter()
        try:
            data = self._call_api(event, snippets)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Defensive extraction — models occasionally omit a field
            raw = data.get("raw_probability") or data.get("probability") or data.get("p")
            if raw is None:
                raise ValueError(
                    f"No probability field in response. Keys present: {list(data.keys())}"
                )
            prob = max(0.01, min(float(raw), 0.99))

            return AgentForecast(
                provider=self.provider_name,
                model=self.model_id,
                reasoning_chain=str(data.get("reasoning_chain", "")),
                key_factors=list(data.get("key_factors", [])),
                raw_probability=prob,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("[%s] Forecast failed: %s", self.provider_name, exc, exc_info=True)
            return AgentForecast(
                provider=self.provider_name,
                model=self.model_id,
                reasoning_chain="",
                key_factors=[],
                raw_probability=0.5,    # neutral fallback; excluded from ensemble mean
                latency_ms=latency_ms,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Anthropic Claude (agent role — Haiku for speed)
# ---------------------------------------------------------------------------

class ClaudeForecaster(BaseForecaster):
    """
    Uses Claude Haiku (fast, cost-effective) as an ensemble agent.
    Claude Sonnet is reserved for the supervisor role.
    Structured output enforced via tool-use.
    """

    _TOOL = {
        "name": "submit_forecast",
        "description": "Submit your structured macroeconomic probability forecast.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning_chain": {"type": "string"},
                "key_factors": {"type": "array", "items": {"type": "string"}},
                "raw_probability": {"type": "number"},
            },
            "required": ["reasoning_chain", "key_factors", "raw_probability"],
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self._client = anthropic.Anthropic(api_key=key)

    @property
    def provider_name(self) -> str:
        return "Anthropic Claude"

    @property
    def model_id(self) -> str:
        return "claude-haiku-4-5"

    def _call_api(self, event: str, snippets: List[NewsSnippet]) -> dict:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            system=_AGENT_SYSTEM_PROMPT,
            tools=[self._TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": _build_user_message(event, snippets)}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_forecast":
                return block.input  # type: ignore[attr-defined]
        raise ValueError("Claude did not call submit_forecast tool.")


# ---------------------------------------------------------------------------
# OpenAI GPT-4o
# ---------------------------------------------------------------------------

class OpenAIForecaster(BaseForecaster):
    """
    OpenAI GPT-4o agent. Uses JSON mode for structured output.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set.")
        self._client = OpenAI(api_key=key)

    @property
    def provider_name(self) -> str:
        return "OpenAI GPT-4o"

    @property
    def model_id(self) -> str:
        return "gpt-4o"

    def _call_api(self, event: str, snippets: List[NewsSnippet]) -> dict:
        response = self._client.chat.completions.create(
            model=self.model_id,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(event, snippets)},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        return _parse_json_response(text, self.provider_name)


# ---------------------------------------------------------------------------
# xAI Grok (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class XAIForecaster(BaseForecaster):
    """
    xAI Grok agent via OpenAI-compatible API at https://api.x.ai/v1.
    Grok's distinct training pipeline provides genuinely different priors
    on geopolitical and market events.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("XAI_API_KEY")
        if not key:
            raise ValueError("XAI_API_KEY not set.")
        self._client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")

    @property
    def provider_name(self) -> str:
        return "xAI Grok"

    @property
    def model_id(self) -> str:
        return "grok-4-latest"

    def _call_api(self, event: str, snippets: List[NewsSnippet]) -> dict:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(event, snippets)},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        return _parse_json_response(text, self.provider_name)


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiForecaster(BaseForecaster):
    """
    Google Gemini 2.0 Flash agent.
    Gemini's long-context pretraining and multimodal architecture yields
    different document-level reasoning patterns vs. other providers.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=_AGENT_SYSTEM_PROMPT,
        )

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    @property
    def model_id(self) -> str:
        return "gemini-2.5-flash"

    def _call_api(self, event: str, snippets: List[NewsSnippet]) -> dict:
        response = self._model.generate_content(
            _build_user_message(event, snippets),
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=4096,  # Gemini is verbose; 1024 truncates mid-JSON
                response_mime_type="application/json",  # forces valid JSON output
            ),
        )
        text = response.text or ""
        return _parse_json_response(text, self.provider_name)


# ---------------------------------------------------------------------------
# Temperature Sampling agent (paper's M=10 method, adapted to M=3)
# ---------------------------------------------------------------------------

class TemperatureSampledForecaster(BaseForecaster):
    """
    Implements the paper's original M-sample ensembling strategy:
    run a single model M times with varying temperatures to generate
    forecast diversity through stochastic sampling.

    The paper uses M=10; we default to M=3 for cost efficiency.
    Temperature range [0.3, 0.7, 1.1] spans focused → exploratory reasoning:
      T=0.3  — near-greedy, highest fidelity to dominant evidence
      T=0.7  — balanced exploration/exploitation
      T=1.1  — high entropy, surfaces minority reasoning paths

    Uses tool-use for structured output, identical to ClaudeForecaster.
    """

    _TOOL = {
        "name": "submit_forecast",
        "description": "Submit your structured macroeconomic probability forecast.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning_chain": {"type": "string"},
                "key_factors": {"type": "array", "items": {"type": "string"}},
                "raw_probability": {"type": "number"},
            },
            "required": ["reasoning_chain", "key_factors", "raw_probability"],
        },
    }

    def __init__(
        self,
        temperature: float,
        run_index: int,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        temperature : float
            Sampling temperature for this agent instance.
        run_index : int
            1-based index label (e.g. 1, 2, 3) for UI display.
        model : str
            Claude model ID to use for all M runs.
        """
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        self._client = anthropic.Anthropic(api_key=key)
        self._temperature = temperature
        self._run_index = run_index
        self._model = model

    @property
    def provider_name(self) -> str:
        return f"Claude Run {self._run_index} (T={self._temperature})"

    @property
    def model_id(self) -> str:
        return self._model

    def _call_api(self, event: str, snippets: List[NewsSnippet]) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_AGENT_SYSTEM_PROMPT,
            tools=[self._TOOL],
            tool_choice={"type": "any"},
            temperature=self._temperature,
            messages=[{"role": "user", "content": _build_user_message(event, snippets)}],
        )
        # Primary path: tool use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_forecast":
                tool_input = block.input  # type: ignore[attr-defined]
                # Validate required field present before returning
                if "raw_probability" in tool_input:
                    return tool_input
                # Field missing — attempt JSON fallback from text blocks below
                logger.warning(
                    "T=%.2f tool input missing raw_probability, keys=%s — trying text fallback",
                    self._temperature, list(tool_input.keys()),
                )

        # Fallback: model may have emitted JSON in a text block despite tool_choice
        for block in response.content:
            if hasattr(block, "text") and block.text:  # type: ignore[attr-defined]
                try:
                    return _parse_json_response(block.text, self.provider_name)  # type: ignore[attr-defined]
                except (ValueError, json.JSONDecodeError):
                    pass

        raise ValueError(
            f"Claude T={self._temperature} did not produce a usable forecast. "
            f"Response blocks: {[b.type for b in response.content]}"
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

# Maps display name → provider class (for cross-model mode)
_PROVIDER_REGISTRY: dict[str, type] = {
    "Claude Haiku":   ClaudeForecaster,
    "OpenAI GPT-4o":  OpenAIForecaster,
    "xAI Grok-4":     XAIForecaster,
    "Google Gemini":  GeminiForecaster,
}


def build_cross_model_agents(
    enabled: Optional[List[str]] = None,
) -> List[BaseForecaster]:
    """
    Build cross-model ensemble agents for the selected providers.

    Parameters
    ----------
    enabled : List[str] | None
        Provider display names to include (e.g. ["Claude Haiku", "OpenAI GPT-4o"]).
        Defaults to all four if None.

    Returns
    -------
    List[BaseForecaster]
        Instantiated agents for all enabled + available providers.
    """
    if enabled is None:
        enabled = list(_PROVIDER_REGISTRY.keys())

    agents: List[BaseForecaster] = []
    for name in enabled:
        cls = _PROVIDER_REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown provider '%s' — skipping.", name)
            continue
        try:
            agents.append(cls())
            logger.info("Registered cross-model agent: %s", name)
        except ValueError as exc:
            logger.warning("Skipping %s — %s", name, exc)

    if not agents:
        raise RuntimeError("No cross-model agents could be initialised. Check API keys.")
    return agents


def build_temperature_agents(
    model: str = "claude-sonnet-4-6",
    m: int = 3,
    temperatures: Optional[List[float]] = None,
) -> List[BaseForecaster]:
    """
    Build M temperature-sampled Claude agents (paper's original method).

    Parameters
    ----------
    model : str
        Claude model ID to use for all M runs.
    m : int
        Number of independent samples (paper uses M=10; we default to M=3).
    temperatures : List[float] | None
        Explicit temperature values. If None, evenly spaced from 0.3 to 1.1.

    Returns
    -------
    List[TemperatureSampledForecaster]
    """
    if temperatures is None:
        if m == 1:
            temperatures = [0.7]
        else:
            step = (1.0 - 0.3) / (m - 1)
            temperatures = [round(0.3 + step * i, 2) for i in range(m)]

    agents: List[BaseForecaster] = []
    for i, temp in enumerate(temperatures, start=1):
        try:
            agents.append(TemperatureSampledForecaster(
                temperature=temp,
                run_index=i,
                model=model,
            ))
            logger.info("Registered temperature agent: %s T=%.2f", model, temp)
        except ValueError as exc:
            logger.warning("Skipping temperature agent T=%.2f — %s", temp, exc)

    if not agents:
        raise RuntimeError("No temperature agents could be initialised. Check ANTHROPIC_API_KEY.")
    return agents


def build_ensemble_agents(
    strategy: str = "cross_model",
    enabled_providers: Optional[List[str]] = None,
    temp_model: str = "claude-sonnet-4-6",
    m: int = 3,
) -> List[BaseForecaster]:
    """
    Unified factory. Builds the agent list based on ensemble strategy.

    Parameters
    ----------
    strategy : str
        One of:
          "cross_model"   — 4 distinct model families (our extension)
          "temperature"   — M Claude runs with varying temperatures (paper method)
          "hybrid"        — both strategies combined
    enabled_providers : List[str] | None
        For cross_model / hybrid: which providers to include.
    temp_model : str
        For temperature / hybrid: which Claude model to temperature-sample.
    m : int
        For temperature / hybrid: number of temperature samples.

    Returns
    -------
    List[BaseForecaster]
    """
    agents: List[BaseForecaster] = []

    if strategy in ("cross_model", "hybrid"):
        agents.extend(build_cross_model_agents(enabled_providers))

    if strategy in ("temperature", "hybrid"):
        agents.extend(build_temperature_agents(model=temp_model, m=m))

    if not agents:
        raise RuntimeError("No agents could be initialised for the selected strategy.")

    logger.info(
        "Ensemble built: strategy=%s n_agents=%d providers=%s",
        strategy, len(agents), [a.provider_name for a in agents],
    )
    return agents
