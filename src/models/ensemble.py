"""
src/models/ensemble.py
-----------------------
Parallel ensemble orchestrator.

Runs all provider agents concurrently via ThreadPoolExecutor (since the
provider SDKs are synchronous and Streamlit's event loop can't host native
asyncio coroutines). Collects results, filters failures, and computes
ensemble statistics.

Paper reference: Section 3 — "Ensembling independent agent forecasts reduces
variance. By Jensen's inequality, for strictly convex losses (Brier score),
the expected loss of the mean forecast is strictly less than the mean of
individual losses."

Our extension: agents use distinct model families → correlated errors are
minimised compared to same-model ensembling with temperature variation.
"""

from __future__ import annotations

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

from src.data.client import NewsSnippet
from src.models.providers import AgentForecast, BaseForecaster

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    """
    Aggregated output of the parallel multi-provider ensemble.

    Fields
    ------
    forecasts          : All AgentForecast objects (including failures)
    mean_probability   : Simple mean of successful forecasts (primary signal)
    std_probability    : Standard deviation across successful forecasts
                         (higher = agents disagree more = supervisor should search harder)
    median_probability : Median of successful forecasts
    successful_agents  : Subset of forecasts that did not error
    failed_agents      : Subset that errored (for UI display)
    """
    forecasts: List[AgentForecast]
    mean_probability: float
    std_probability: float
    median_probability: float
    successful_agents: List[AgentForecast] = field(default_factory=list)
    failed_agents: List[AgentForecast] = field(default_factory=list)

    @property
    def n_successful(self) -> int:
        return len(self.successful_agents)

    @property
    def n_failed(self) -> int:
        return len(self.failed_agents)

    @property
    def agent_disagreement_level(self) -> str:
        """Qualitative label for how much agents disagree (maps to supervisor search intensity)."""
        if self.std_probability >= 0.15:
            return "High"
        elif self.std_probability >= 0.08:
            return "Moderate"
        return "Low"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class EnsembleOrchestrator:
    """
    Runs all registered BaseForecaster agents in parallel and returns an
    EnsembleResult with per-agent outputs and aggregate statistics.

    Parameters
    ----------
    agents : List[BaseForecaster]
        Provider agents to run (built by providers.build_ensemble_agents()).
    max_workers : int
        Thread pool size. Should equal len(agents) so all run simultaneously.
    """

    def __init__(self, agents: List[BaseForecaster], max_workers: int = 6):
        self.agents = agents
        self.max_workers = max_workers

    def run(self, event: str, snippets: List[NewsSnippet]) -> EnsembleResult:
        """
        Submit all agent forecast tasks in parallel and collect results.

        Parameters
        ----------
        event : str
            The binary macro event being forecast.
        snippets : List[NewsSnippet]
            Context snippets from MacroDataFetcher.

        Returns
        -------
        EnsembleResult
            Aggregate statistics and per-agent forecasts.

        Raises
        ------
        RuntimeError
            If fewer than 2 agents succeed (insufficient for meaningful ensemble).
        """
        forecasts: List[AgentForecast] = []

        logger.info(
            "Starting parallel ensemble: %d agents for event: %s",
            len(self.agents), event,
        )

        # Submit all agents concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(agent.forecast, event, snippets): agent
                for agent in self.agents
            }
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result: AgentForecast = future.result()
                    forecasts.append(result)
                    status = "OK" if result.succeeded else f"FAILED: {result.error}"
                    logger.info(
                        "[%s] %s — prob=%.3f latency=%.0fms",
                        agent.provider_name, status,
                        result.raw_probability, result.latency_ms,
                    )
                except Exception as exc:
                    # Future itself raised — shouldn't happen since BaseForecaster
                    # catches internally, but defensive handling
                    logger.error("[%s] Future raised: %s", agent.provider_name, exc)
                    forecasts.append(AgentForecast(
                        provider=agent.provider_name,
                        model=agent.model_id,
                        reasoning_chain="",
                        key_factors=[],
                        raw_probability=0.5,
                        latency_ms=0.0,
                        error=str(exc),
                    ))

        # Partition into successful and failed
        successful = [f for f in forecasts if f.succeeded]
        failed = [f for f in forecasts if not f.succeeded]

        if len(successful) < 2:
            raise RuntimeError(
                f"Only {len(successful)} agent(s) succeeded. "
                "Check API keys and network connectivity."
            )

        probs = [f.raw_probability for f in successful]
        mean_p = statistics.mean(probs)
        std_p = statistics.stdev(probs) if len(probs) > 1 else 0.0
        median_p = statistics.median(probs)

        logger.info(
            "Ensemble complete: n=%d mean=%.4f std=%.4f median=%.4f",
            len(successful), mean_p, std_p, median_p,
        )

        return EnsembleResult(
            forecasts=forecasts,
            mean_probability=mean_p,
            std_probability=std_p,
            median_probability=median_p,
            successful_agents=successful,
            failed_agents=failed,
        )
