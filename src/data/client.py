"""
src/data/client.py
------------------
Agentic data ingestion layer.

MacroDataFetcher simulates retrieval of recent macroeconomic text data for a
given event query. In a production system this would hit live data providers
(Bloomberg, Fed FRED API, Reuters, etc.). Here we use a deterministic but
realistic simulation so the app runs without external data subscriptions.

The snippets are intentionally slightly contradictory — mirroring real-world
information environments where market participants read different signals from
the same macro data.
"""

from __future__ import annotations

import random
import textwrap
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NewsSnippet:
    """A single piece of macro context (headline + body + source + date)."""
    source: str
    date: str          # e.g. "2025-03-05"
    headline: str
    body: str

    def __str__(self) -> str:
        return (
            f"[{self.source} | {self.date}]\n"
            f"**{self.headline}**\n"
            f"{self.body}"
        )


# ---------------------------------------------------------------------------
# Snippet libraries — realistic templates keyed by macro theme
# ---------------------------------------------------------------------------

# Each theme maps to a list of snippet templates. {event} is replaced at
# runtime with the user's query so results feel contextually grounded.
_SNIPPET_TEMPLATES: dict[str, list[dict]] = {

    # ---- Federal Reserve / interest rates ---------------------------------
    "fed_rates": [
        {
            "source": "Federal Reserve Press Release",
            "date": "2025-03-05",
            "headline": "FOMC holds benchmark rate steady; signals patience on cuts",
            "body": (
                "The Federal Open Market Committee voted unanimously to maintain "
                "the federal funds rate in the 5.25–5.50% target range. Chair "
                "Powell stated the committee 'does not expect it will be appropriate "
                "to reduce the target range until it has gained greater confidence "
                "that inflation is moving sustainably toward 2 percent.'"
            ),
        },
        {
            "source": "Wall Street Journal — Markets Desk",
            "date": "2025-03-06",
            "headline": "Fed officials split on timing of first rate cut",
            "body": (
                "Internal Fed minutes revealed a debate between hawks who see "
                "inflation risks as unresolved and doves pointing to softening "
                "labor demand. Two governors suggested a June cut remains viable "
                "if PCE prints come in below 2.5% in April and May."
            ),
        },
        {
            "source": "Bloomberg Economics",
            "date": "2025-03-07",
            "headline": "Markets price 68% probability of 25bps cut in May",
            "body": (
                "Fed-funds futures moved sharply after Friday's jobs report showed "
                "payrolls growth slowing to 142k vs. 185k expected. The implied "
                "probability of a May cut rose from 41% to 68% overnight. Several "
                "primary dealers revised their base cases forward by one meeting."
            ),
        },
        {
            "source": "Reuters — Global Economy",
            "date": "2025-03-07",
            "headline": "Sticky services inflation complicates Fed's easing calculus",
            "body": (
                "Core services ex-housing — the metric Powell has called 'the most "
                "important category for understanding the future evolution of core "
                "inflation' — remained stubbornly above 4% annualised for the third "
                "consecutive month. Analysts say this limits the Fed's room to cut "
                "without risking a credibility hit."
            ),
        },
        {
            "source": "Goldman Sachs Global Investment Research",
            "date": "2025-03-08",
            "headline": "GS upgrades Fed cut forecast to 3 cuts in 2025",
            "body": (
                "Goldman Sachs economists revised their FOMC call, now forecasting "
                "three 25bp cuts starting in June, citing a faster-than-expected "
                "cooling in the labour market and improving supply chains. They "
                "assign a 30% probability to an earlier May cut."
            ),
        },
        {
            "source": "Morgan Stanley Research",
            "date": "2025-03-08",
            "headline": "MS maintains 'higher-for-longer' view through Q3 2025",
            "body": (
                "Contrary to peers, Morgan Stanley's rate strategy team argues the "
                "Fed will stay on hold through Q3. 'The last mile of disinflation "
                "is proving far more difficult,' the note reads. They see only one "
                "cut in 2025 with risk skewed toward no cuts at all."
            ),
        },
        {
            "source": "Fed Governor Christopher Waller — Speech, Stanford",
            "date": "2025-03-04",
            "headline": "Waller: 'I need a few more months of good data before supporting cuts'",
            "body": (
                "Governor Waller pushed back on market exuberance, saying he "
                "personally requires two to three more months of inflation data "
                "trending toward target before he would feel comfortable voting "
                "to lower the policy rate. He characterised current financial "
                "conditions as 'appropriately restrictive.'"
            ),
        },
    ],

    # ---- Recession / GDP --------------------------------------------------
    "recession": [
        {
            "source": "Bureau of Economic Analysis — Advance Estimate",
            "date": "2025-03-05",
            "headline": "Q4 GDP revised down to 1.4% annualised; below consensus 1.8%",
            "body": (
                "The BEA's second estimate trimmed Q4 GDP growth to 1.4% from the "
                "advance reading of 1.7%. Net exports subtracted 0.8pp while "
                "residential investment added 0.3pp. Several economists flagged "
                "that the underlying domestic demand figure remains relatively firm."
            ),
        },
        {
            "source": "Conference Board — Leading Economic Index",
            "date": "2025-03-06",
            "headline": "LEI falls for 6th consecutive month, deepening recession fears",
            "body": (
                "The Conference Board's Leading Economic Index declined 0.4% in "
                "February, the sixth consecutive monthly decline. The organisation's "
                "chief economist stated the pattern is 'consistent with recession "
                "in the next 12 months' though acknowledged the signal has shown "
                "false positives in recent cycles."
            ),
        },
        {
            "source": "JPMorgan Asset Management — Macro Outlook",
            "date": "2025-03-07",
            "headline": "JPM raises US recession probability to 35% for next 12 months",
            "body": (
                "JPMorgan raised its 12-month US recession probability from 25% to "
                "35%, citing tighter credit conditions for small businesses, lagged "
                "effects of monetary tightening, and a weakening consumer balance "
                "sheet as pandemic-era savings are drawn down."
            ),
        },
        {
            "source": "Oxford Economics",
            "date": "2025-03-07",
            "headline": "Labour market resilience argues against near-term recession",
            "body": (
                "Oxford Economics' US team maintains a soft-landing base case, "
                "noting that unemployment remains below 4.2% and real wage growth "
                "has turned positive. 'A recession without a significant labour "
                "market deterioration would be historically anomalous,' their "
                "report concludes."
            ),
        },
        {
            "source": "NBER Business Cycle Dating Committee — Statement",
            "date": "2025-02-28",
            "headline": "NBER: No recession has been declared; monitoring conditions",
            "body": (
                "The NBER's BCDC issued a routine statement confirming no current "
                "recession declaration and emphasising that monthly indicators "
                "including employment, personal income, and industrial production "
                "remain above their pre-pandemic trend levels."
            ),
        },
        {
            "source": "Deutsche Bank Research",
            "date": "2025-03-08",
            "headline": "DB flags credit card delinquency surge as recession early warning",
            "body": (
                "Deutsche Bank analysts highlighted that credit card delinquency "
                "rates have reached their highest level since 2011, with the "
                "90-day delinquency rate up 45bps year-over-year. 'This is the "
                "canary in the coal mine for consumer stress,' their note warns."
            ),
        },
    ],

    # ---- Inflation / CPI --------------------------------------------------
    "inflation": [
        {
            "source": "Bureau of Labor Statistics — CPI Release",
            "date": "2025-03-05",
            "headline": "February CPI +3.1% YoY; shelter inflation remains elevated",
            "body": (
                "The Consumer Price Index rose 3.1% year-over-year in February, "
                "in line with consensus. Core CPI came in at 3.8% YoY. Shelter "
                "inflation contributed 1.9pp of the total, while energy prices "
                "fell 1.2%. Owners' Equivalent Rent is showing early signs of "
                "moderation but remains above pre-pandemic norms."
            ),
        },
        {
            "source": "Cleveland Fed — Inflation Nowcast",
            "date": "2025-03-06",
            "headline": "Nowcast signals March CPI likely to come in at 2.9%",
            "body": (
                "The Cleveland Fed's real-time Inflation Nowcast model projects "
                "March CPI at 2.9% YoY — the first sub-3% reading since early "
                "2021 — driven primarily by base effects from last year's energy "
                "spike and a continued softening in used-vehicle prices."
            ),
        },
        {
            "source": "UBS Global Macro Research",
            "date": "2025-03-07",
            "headline": "UBS warns of inflation re-acceleration risk in H2 2025",
            "body": (
                "Despite the recent CPI moderation, UBS's macro team identifies "
                "three upside risks: (1) geopolitical oil supply disruptions, "
                "(2) tariff pass-through from new trade policy, and (3) a "
                "re-acceleration of wage growth if the labour market tightens. "
                "They assign a 25% probability to CPI re-accelerating above 4% "
                "by year-end."
            ),
        },
        {
            "source": "University of Michigan — Consumer Sentiment Survey",
            "date": "2025-03-08",
            "headline": "Long-run inflation expectations tick up to 3.0%; 10-year high",
            "body": (
                "Consumers now expect inflation to average 3.0% over the next "
                "5–10 years, up from 2.9% last month and the highest reading in "
                "a decade. The Fed views de-anchored long-run expectations as a "
                "serious risk, as they can become self-fulfilling through wage "
                "bargaining and pricing behaviour."
            ),
        },
        {
            "source": "Citi Economic Research",
            "date": "2025-03-08",
            "headline": "Citi maintains disinflation trend is intact; sees 2% by year-end",
            "body": (
                "Citi's economists remain constructive on disinflation, forecasting "
                "headline CPI will reach 2.0% by December 2025. Key drivers: "
                "a 12-month lag on shelter CPI catching up to real-time rent data, "
                "normalising goods prices, and muted oil demand from China."
            ),
        },
        {
            "source": "Fed Beige Book — March 2025",
            "date": "2025-03-05",
            "headline": "Beige Book: Most districts report 'modest' price pressures",
            "body": (
                "The March Beige Book noted that most of the Fed's twelve districts "
                "reported 'modest' or 'moderate' input cost increases. Several "
                "retailers indicated reduced pricing power as consumers pushed back "
                "on further price increases, suggesting demand-side inflation is "
                "cooling even as services costs remain sticky."
            ),
        },
    ],

    # ---- Generic fallback — used when none of the above themes match ------
    "generic": [
        {
            "source": "IMF World Economic Outlook — Update",
            "date": "2025-03-05",
            "headline": "IMF upgrades global growth forecast to 3.2% but flags risks",
            "body": (
                "The IMF revised its 2025 global growth forecast up 0.1pp to 3.2%, "
                "driven by stronger-than-expected US momentum. However, the Fund "
                "warned of downside risks from geopolitical fragmentation, elevated "
                "debt levels, and the lagged effects of global monetary tightening."
            ),
        },
        {
            "source": "World Bank — Global Economic Prospects",
            "date": "2025-03-06",
            "headline": "Emerging market debt distress escalating, says World Bank",
            "body": (
                "The World Bank flagged that 60% of low-income countries are at "
                "high risk of debt distress, complicating the global outlook. "
                "Higher-for-longer rates in developed markets are tightening "
                "financing conditions for emerging economies and threatening "
                "development goals."
            ),
        },
        {
            "source": "OECD Economic Outlook — Interim Report",
            "date": "2025-03-07",
            "headline": "OECD: Divergent monetary cycles create FX volatility risk",
            "body": (
                "The OECD's interim outlook highlighted risks from asynchronous "
                "central bank cycles — the Fed holding while the ECB and BoE cut — "
                "creating potential for sharp currency moves. The report urged "
                "policymakers to communicate clearly to avoid market dislocations."
            ),
        },
        {
            "source": "Bank for International Settlements — Quarterly Review",
            "date": "2025-03-05",
            "headline": "BIS warns of 'liquidity illusion' in credit markets",
            "body": (
                "The BIS flagged that tight credit spreads may be masking underlying "
                "fragility, noting that market liquidity can 'evaporate rapidly' in "
                "stress scenarios. The report pointed to the rise of private credit "
                "markets as an area of particular opacity."
            ),
        },
        {
            "source": "Brookings Institution — Economic Studies",
            "date": "2025-03-07",
            "headline": "Fiscal dominance concerns resurface amid widening US deficit",
            "body": (
                "Brookings researchers published analysis suggesting the US primary "
                "deficit trajectory could begin constraining Fed independence within "
                "a decade. The paper argues that without fiscal consolidation, the "
                "central bank may face pressure to keep rates lower than inflation "
                "dynamics alone would warrant."
            ),
        },
        {
            "source": "BlackRock Investment Institute",
            "date": "2025-03-08",
            "headline": "BlackRock: Macro regime has fundamentally shifted; new playbook needed",
            "body": (
                "BlackRock's macro team argues that the post-GFC era of low rates "
                "and low volatility is permanently over, replaced by a 'new regime' "
                "of higher structural inflation, greater geopolitical risk, and "
                "more volatile growth cycles. They recommend shorter duration "
                "assets and real assets as portfolio hedges."
            ),
        },
        {
            "source": "Federal Reserve Bank of New York — Survey of Consumer Expectations",
            "date": "2025-03-08",
            "headline": "NY Fed: 1-year ahead inflation expectations fall to 3.0%",
            "body": (
                "The NY Fed's monthly survey showed 1-year-ahead inflation "
                "expectations declined 0.2pp to 3.0%, the lowest since March 2021. "
                "However, income growth expectations also fell, suggesting consumers "
                "see a softening economic backdrop ahead."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Helper: theme detection
# ---------------------------------------------------------------------------

def _detect_theme(event: str) -> str:
    """
    Map a free-text event query to one of our snippet library themes.
    Simple keyword matching is sufficient for the prototype; a production
    system would use an embedding-based classifier.
    """
    event_lower = event.lower()

    if any(kw in event_lower for kw in ["rate", "fed", "fomc", "bps", "cut", "hike", "pivot"]):
        return "fed_rates"
    if any(kw in event_lower for kw in ["recession", "gdp", "growth", "contraction", "downturn"]):
        return "recession"
    if any(kw in event_lower for kw in ["inflation", "cpi", "pce", "price", "deflation"]):
        return "inflation"
    return "generic"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MacroDataFetcher:
    """
    Simulates an agentic data ingestion pipeline that retrieves recent
    macroeconomic news and official statements relevant to a given event.

    In production this class would:
      - Query live APIs (Bloomberg Terminal, FRED, Refinitiv)
      - Scrape central bank websites and press releases
      - Run semantic search over an internal document store
      - Deduplicate and rank snippets by relevance score

    For the prototype we return deterministic, realistic simulated data.
    """

    def __init__(self, num_snippets: int = 6, random_seed: int | None = None):
        """
        Parameters
        ----------
        num_snippets : int
            How many context snippets to return (5–7 recommended).
        random_seed : int | None
            Set for reproducible output during testing.
        """
        self.num_snippets = max(5, min(num_snippets, 7))  # clamp to 5–7
        if random_seed is not None:
            random.seed(random_seed)

    def fetch_recent_context(self, event: str) -> List[NewsSnippet]:
        """
        Retrieve recent macroeconomic context snippets for a given event query.

        Parameters
        ----------
        event : str
            Natural-language description of the macro event to forecast
            (e.g. "Will the Fed cut rates by 25bps at the next FOMC meeting?").

        Returns
        -------
        List[NewsSnippet]
            A list of 5–7 slightly conflicting news snippets and official
            statements providing context for the LLM reasoning agent.
        """
        theme = _detect_theme(event)
        pool = _SNIPPET_TEMPLATES[theme]

        # If we need more snippets than the themed pool has, supplement with
        # generic snippets (excluding any already in the themed pool).
        if len(pool) < self.num_snippets:
            generic_pool = _SNIPPET_TEMPLATES["generic"]
            supplement = [s for s in generic_pool if s not in pool]
            pool = pool + supplement

        # Sample without replacement so we don't repeat snippets
        selected_raw = random.sample(pool, min(self.num_snippets, len(pool)))

        # Convert dicts to typed NewsSnippet objects
        snippets: List[NewsSnippet] = []
        for raw in selected_raw:
            snippets.append(
                NewsSnippet(
                    source=raw["source"],
                    date=raw["date"],
                    headline=raw["headline"],
                    body=textwrap.dedent(raw["body"]).strip(),
                )
            )

        return snippets
