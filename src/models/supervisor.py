"""
src/models/supervisor.py
-------------------------
Claude Sonnet Supervisor Agent.

The supervisor is the most critical novel contribution of the AIA paper
(Section 4.3). Unlike naive aggregation (which performs *worse* than simple
mean — LLMs over-weight outlier opinions), the supervisor:

  1. Reads all agent reasoning traces side-by-side
  2. Identifies SPECIFIC points of disagreement (not just probability spread)
  3. Issues targeted clarifying queries (simulated here) to resolve them
  4. Confidence-gates its output:
       - high confidence   → supervisor probability replaces ensemble mean
       - medium/low conf.  → ensemble mean is preserved

Key insight from the paper: reframing from "synthesise opinions" to "find and
resolve specific evidentiary disagreements" is what makes the supervisor work.
Asking an LLM to merely average opinions is provably worse than the simple mean.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import anthropic

from src.models.ensemble import EnsembleResult
from src.models.providers import AgentForecast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SupervisorResult:
    """
    Output of the Claude supervisor reconciliation step.

    Fields
    ------
    disagreements_identified : Specific claims where agents diverged
    reconciliation_reasoning : Supervisor's chain-of-thought
    reconciled_probability   : Supervisor's proposed probability
    confidence               : "high" | "medium" | "low"
    key_evidence_gaps        : Unresolved information gaps
    final_probability        : The value to carry forward:
                                 - reconciled_probability if confidence == "high"
                                 - ensemble mean otherwise
    used_supervisor_output   : True if high-confidence override occurred
    """
    disagreements_identified: List[str]
    reconciliation_reasoning: str
    reconciled_probability: float
    confidence: str
    key_evidence_gaps: List[str]
    ensemble_mean: float
    final_probability: float
    used_supervisor_output: bool


# ---------------------------------------------------------------------------
# Supervisor tool schema
# ---------------------------------------------------------------------------

_SUPERVISOR_TOOL = {
    "name": "submit_reconciliation",
    "description": (
        "Submit your reconciled forecast after analysing all agent reasoning traces. "
        "Only call this once — after you have identified all disagreements."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "disagreements_identified": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Each entry should name a SPECIFIC factual or interpretive claim "
                    "where at least two agents took clearly different positions. "
                    "E.g. 'Agent A says inflation is falling; Agent B says it is re-accelerating.'"
                ),
            },
            "reconciliation_reasoning": {
                "type": "string",
                "description": (
                    "Your detailed reasoning for the reconciled probability. "
                    "Explain which evidence you found most reliable and why, "
                    "and how you resolved each identified disagreement."
                ),
            },
            "reconciled_probability": {
                "type": "number",
                "description": (
                    "Your reconciled probability estimate in [0.0, 1.0]. "
                    "This replaces the ensemble mean ONLY if confidence is 'high'."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "Your confidence in this reconciliation. "
                    "'high': you found clear evidentiary resolution → your estimate replaces mean. "
                    "'medium' or 'low': agents' disagreement is unresolvable → ensemble mean is kept."
                ),
            },
            "key_evidence_gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Information gaps that prevented full resolution of disagreements.",
            },
        },
        "required": [
            "disagreements_identified",
            "reconciliation_reasoning",
            "reconciled_probability",
            "confidence",
            "key_evidence_gaps",
        ],
    },
}

# ---------------------------------------------------------------------------
# Supervisor system prompt
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM = """\
You are the AIA-Macro Supervisor. You have received independent probability \
forecasts from multiple AI agents (Claude, GPT-4o, Grok, Gemini) on a \
macroeconomic event. Your role is NOT to average their opinions — that is \
already done. Your role is to act as an investigative analyst.

## Your Task (in order)
1. Read every agent's reasoning trace carefully.
2. Identify SPECIFIC factual claims or interpretations where agents disagree.
3. For each disagreement, determine which agent's evidence is stronger and why.
4. Produce a reconciled probability that reflects the resolved evidence.

## Critical Rules
- Do NOT simply average the probabilities — that is already computed.
- Do NOT defer to the agent with the most extreme view.
- Do NOT produce a "compromise" unless the evidence is genuinely balanced.
- ONLY assign 'high' confidence if you found specific evidentiary grounds to
  override the ensemble mean. If agents simply interpreted the same ambiguous
  evidence differently, assign 'medium' or 'low'.

## Output
Call the `submit_reconciliation` tool exactly once.
"""


# ---------------------------------------------------------------------------
# Supervisor class
# ---------------------------------------------------------------------------

class SupervisorAgent:
    """
    Claude Sonnet-powered supervisor that reads all ensemble reasoning traces,
    identifies specific disagreements, and confidence-gates its output.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model : str
        Claude model to use as supervisor. Defaults to claude-sonnet-4-6
        (stronger reasoning than Haiku, reserved specifically for this role).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set for supervisor.")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model

    def _build_supervisor_prompt(
        self,
        event: str,
        ensemble: EnsembleResult,
    ) -> str:
        """
        Construct the user turn for the supervisor: event + all agent traces.
        """
        lines = [
            f"## Event\n\n{event}\n",
            f"## Ensemble Statistics\n",
            f"- Simple mean probability: **{ensemble.mean_probability:.4f}**",
            f"- Standard deviation: **{ensemble.std_probability:.4f}** "
            f"({ensemble.agent_disagreement_level} disagreement)",
            f"- Median: **{ensemble.median_probability:.4f}**",
            f"- Successful agents: {ensemble.n_successful}\n",
            "---\n",
            "## Individual Agent Forecasts\n",
        ]

        for i, agent in enumerate(ensemble.successful_agents, start=1):
            lines.append(
                f"### Agent {i}: {agent.provider} ({agent.model})\n"
                f"**Probability:** {agent.raw_probability:.4f}\n\n"
                f"**Reasoning:**\n{agent.reasoning_chain}\n\n"
                f"**Key Factors:**\n"
                + "\n".join(f"- {f}" for f in agent.key_factors)
                + "\n\n---\n"
            )

        lines.append(
            "Now identify specific disagreements between agents, resolve them "
            "where possible, and call `submit_reconciliation`."
        )

        return "\n".join(lines)

    def reconcile(self, event: str, ensemble: EnsembleResult) -> SupervisorResult:
        """
        Run the supervisor reconciliation step.

        Parameters
        ----------
        event : str
            The macro event being forecast.
        ensemble : EnsembleResult
            Output of the ensemble orchestrator.

        Returns
        -------
        SupervisorResult
            Includes the reconciled probability, confidence level, and the
            final_probability to be passed to calibration.
        """
        user_message = self._build_supervisor_prompt(event, ensemble)

        logger.info(
            "Supervisor analysing %d agent traces (disagreement: %s)",
            ensemble.n_successful,
            ensemble.agent_disagreement_level,
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SUPERVISOR_SYSTEM,
            tools=[_SUPERVISOR_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract tool use block
        tool_input: Optional[dict] = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_reconciliation":
                tool_input = block.input  # type: ignore[attr-defined]
                break

        if tool_input is None:
            logger.warning("Supervisor did not call tool — deferring to ensemble mean.")
            return SupervisorResult(
                disagreements_identified=[],
                reconciliation_reasoning="Supervisor did not produce a tool call; deferring to ensemble mean.",
                reconciled_probability=ensemble.mean_probability,
                confidence="low",
                key_evidence_gaps=["Supervisor call failed"],
                ensemble_mean=ensemble.mean_probability,
                final_probability=ensemble.mean_probability,
                used_supervisor_output=False,
            )

        # Confidence gating: only override mean if supervisor is high-confidence
        confidence = tool_input.get("confidence", "low")
        reconciled_p = float(tool_input.get("reconciled_probability", ensemble.mean_probability))
        reconciled_p = max(0.01, min(reconciled_p, 0.99))

        used_supervisor = confidence == "high"
        final_p = reconciled_p if used_supervisor else ensemble.mean_probability

        logger.info(
            "Supervisor: confidence=%s reconciled=%.4f final=%.4f (used_supervisor=%s)",
            confidence, reconciled_p, final_p, used_supervisor,
        )

        return SupervisorResult(
            disagreements_identified=tool_input.get("disagreements_identified", []),
            reconciliation_reasoning=tool_input.get("reconciliation_reasoning", ""),
            reconciled_probability=reconciled_p,
            confidence=confidence,
            key_evidence_gaps=tool_input.get("key_evidence_gaps", []),
            ensemble_mean=ensemble.mean_probability,
            final_probability=final_p,
            used_supervisor_output=used_supervisor,
        )
