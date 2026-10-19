"""Headline sentiment with VADER, run locally. No LLM call, no network.

VADER is a general-purpose lexicon; a few crypto words are added so "bullish", "hack" or
"liquidations" score the way a trader reads them. Scores are VADER's compound value, -1..+1.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

CRYPTO_LEXICON = {
    "bullish": 2.0,
    "bearish": -2.0,
    "rally": 1.5,
    "rallies": 1.5,
    "surge": 1.5,
    "surges": 1.5,
    "soar": 1.8,
    "soars": 1.8,
    "breakout": 1.2,
    "ath": 1.5,
    "inflows": 1.0,
    "outflows": -1.0,
    "plunge": -2.0,
    "plunges": -2.0,
    "crash": -2.5,
    "crashes": -2.5,
    "dump": -1.8,
    "dumps": -1.8,
    "selloff": -1.8,
    "sell-off": -1.8,
    "hack": -2.5,
    "hacked": -2.5,
    "exploit": -2.5,
    "exploited": -2.5,
    "liquidations": -1.5,
    "liquidated": -1.5,
    "lawsuit": -1.5,
    "sues": -1.5,
    "ban": -1.8,
    "bans": -1.8,
    "delist": -1.8,
    "delisting": -1.8,
    "insolvent": -2.5,
    "bankruptcy": -2.5,
    "approval": 1.2,
    "approved": 1.2,
    "adoption": 1.0,
}

_analyzer = None


def analyzer():
    """The shared VADER analyser, or None if vaderSentiment is not installed."""
    global _analyzer
    if _analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:  # pragma: no cover - it is a declared dependency
            log.warning("vaderSentiment is not installed: headline sentiment is off")
            return None
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(CRYPTO_LEXICON)
    return _analyzer


def score(text: str) -> float | None:
    a = analyzer()
    if a is None or not text:
        return None
    return float(a.polarity_scores(text)["compound"])
