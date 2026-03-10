"""
src/models/reasoning_agent.py
------------------------------
LLM Supervisor — Anthropic-powered macroeconomic reasoning agent.

This module defines:
  - ForecastResult  : Pydantic schema for structured LLM output
  - MacroReasoningAgent : wraps the Anthropic API and enforces structured output
                          via tool-use / JSON extraction.

Design notes
------------
We use Anthropic's tool-use feature to force the model to emit a JSON object
that exactly matches the ForecastResult schema. This is more reliable than
asking the model to produce JSON in its text response and then parsing it.

The system prompt positions the model as a Bayesian macro forecaster who is
explicitly aware of LLM overconfidence biases and instructed to give
well-calibrated raw probabilities — even though our calibration layer will
further correct them downstream.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from src.data.client import NewsSnippet

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic schema for structured output
# ---------------------------------------------------------------------------

class ForecastResult(BaseModel):
    """
    Structured output of the LLM reasoning agent.

    Fields
    ------
    reasoning_chain : str
        Step-by-step chain-of-thought analysis produced by the model, showing
        how it weighed the evidence and arrived at its probability estimate.
    key_factors : list[str]
        Bulleted list of the 3–6 most important factors driving the forecast
        (both supportive and contrary).
    raw_probability : float
        The model's raw probability estimate in [0.0, 1.0] that the event
        will occur. This is BEFORE statistical calibration.
    """

    reasoning_chain: str = Field(
        ...,
        description=(
            "A detailed chain-of-thought analysis explaining how the model "
            "weighed the provided evidence to arrive at its probability estimate."
        ),
    )
    key_factors: List[str] = Field(
        ...,
        min_length=2,
        description="3 to 6 key factors (bulleted) driving the forecast.",
    )
    raw_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that the event occurs, as a float in [0.0, 1.0]. "
            "Must be calibrated carefully — avoid extreme values without strong evidence."
        ),
    )

    @field_validator("raw_probability")
    @classmethod
    def clamp_probability(cls, v: float) -> float:
        """Hard-clamp to [0.01, 0.99] to avoid degenerate edge cases."""
        return max(0.01, min(v, 0.99))


# ---------------------------------------------------------------------------
# Tool schema — used to force structured JSON output from the model
# ---------------------------------------------------------------------------

_FORECAST_TOOL: dict = {
    "name": "submit_forecast",
    "description": (
        "Submit a structured macroeconomic probability forecast. "
        "Call this tool once you have completed your analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning_chain": {
                "type": "string",
                "description": (
                    "Detailed chain-of-thought analysis. Walk through each piece "
                    "of evidence, identify conflicts, and explain how you weigh them."
                ),
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3 to 6 key factors that most influence your forecast.",
            },
            "raw_probability": {
                "type": "number",
                "description": (
                    "Your probability estimate (0.0–1.0) that the event occurs. "
                    "Be thoughtful — LLMs tend to be overconfident."
                ),
            },
        },
        "required": ["reasoning_chain", "key_factors", "raw_probability"],
    },
}

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are AIA-Macro, an objective macroeconomic probability forecaster built by \
the AIA Research Institute. Your role is to produce well-calibrated probability \
estimates for specific macroeconomic events based solely on the evidence provided \
to you.

## Your Mandate
1. Read the provided news snippets and official statements carefully.
2. Identify where sources agree, where they conflict, and why.
3. Reason in the Bayesian tradition: start from a prior, update on evidence.
4. Produce a probability estimate for the stated event.

## Calibration Discipline
- LLMs are systematically overconfident. Resist the pull toward extreme values.
- If evidence is mixed or contradictory, your probability should reflect that \
  uncertainty — typical values in ambiguous cases should be in the 0.35–0.65 range.
- Only justify probabilities above 0.80 or below 0.20 with very strong, \
  consistent evidence.
- Consider base rates: most contested macro events have base rates near 50%.

## Output Format
Use the `submit_forecast` tool to return your structured forecast. \
Do NOT return plain text — only the tool call.
"""

# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class MacroReasoningAgent:
    """
    Wraps the Anthropic API to produce structured macroeconomic forecasts.

    Usage
    -----
    >>> agent = MacroReasoningAgent()
    >>> result = agent.forecast(event="Will the Fed cut rates 25bps?", snippets=snippets)
    >>> print(result.raw_probability)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        api_key: str | None = None,
    ):
        """
        Parameters
        ----------
        model : str
            Anthropic model ID. Defaults to claude-sonnet-4-6 for cost/quality balance.
        max_tokens : int
            Max tokens for the model response.
        api_key : str | None
            Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        """
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in your .env file or pass api_key= "
                "to MacroReasoningAgent()."
            )

        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(event: str, snippets: List[NewsSnippet]) -> str:
        """
        Construct the user-turn message that injects the event query and all
        retrieved context snippets.
        """
        snippet_block = "\n\n---\n\n".join(str(s) for s in snippets)

        return (
            f"## Event to Forecast\n\n"
            f"{event}\n\n"
            f"## Retrieved Context Snippets\n\n"
            f"{snippet_block}\n\n"
            f"---\n\n"
            f"Now produce your probability forecast using the `submit_forecast` tool."
        )

    @staticmethod
    def _extract_tool_input(response: anthropic.types.Message) -> dict:
        """
        Pull the tool-use input dict from the Anthropic response.
        Raises ValueError if no tool_use block is found (shouldn't happen with
        tool_choice='any', but defensive coding is good practice).
        """
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_forecast":
                return block.input  # type: ignore[attr-defined]

        # Fallback: try to parse JSON from text (graceful degradation)
        for block in response.content:
            if hasattr(block, "text"):
                try:
                    return json.loads(block.text)  # type: ignore[attr-defined]
                except json.JSONDecodeError:
                    pass

        raise ValueError(
            "Model did not call the submit_forecast tool. "
            f"Raw response: {response}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forecast(self, event: str, snippets: List[NewsSnippet]) -> ForecastResult:
        """
        Run the LLM reasoning pipeline and return a validated ForecastResult.

        Parameters
        ----------
        event : str
            The macro event to forecast (natural language question).
        snippets : List[NewsSnippet]
            Context snippets from MacroDataFetcher.

        Returns
        -------
        ForecastResult
            Validated Pydantic model with reasoning_chain, key_factors,
            and raw_probability.

        Raises
        ------
        anthropic.APIError
            If the Anthropic API call fails.
        ValueError
            If the model output cannot be parsed into ForecastResult.
        """
        user_message = self._build_user_message(event, snippets)

        logger.info("Sending forecast request to %s for event: %s", self.model, event)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM_PROMPT,
                tools=[_FORECAST_TOOL],
                # Force the model to use our tool — prevents free-text fallback
                tool_choice={"type": "any"},
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API error: %s", exc)
            raise
        except anthropic.APIConnectionError as exc:
            logger.error("Anthropic connection error: %s", exc)
            raise

        # Extract structured output
        raw_input = self._extract_tool_input(response)

        logger.info("Raw LLM probability: %.3f", raw_input.get("raw_probability", -1))

        # Validate through Pydantic (enforces types + clamping validator)
        result = ForecastResult(**raw_input)
        return result
