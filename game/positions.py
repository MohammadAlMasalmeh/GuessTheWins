"""
Infer G / F / C buckets from box-score shape.

`players.primary_position` is mostly empty (bios often skipped), so the game
derives a lineup bucket from the player's highest-minutes qualifying season
stats. Tuned so true bigs (Duncan, Shaq, Gobert) are never labeled G.
"""

from __future__ import annotations

from typing import Optional


def classify_position(
    ast_pct: Optional[float],
    trb_pct: Optional[float],
    blk_pct: Optional[float],
    three_par: Optional[float],
    primary_role: Optional[str] = None,
) -> str:
    """
    Return "G", "F", or "C".

    Order matters: creators without big-man rebounding → G first; then
    clear centers; then forwards; then leftover shooters/creators → G.
    """
    # Role fallback when advanced rates are missing (common pre-1974).
    if ast_pct is None and trb_pct is None and blk_pct is None:
        if primary_role in ("ROLL_BIG", "STRETCH_BIG"):
            return "C"
        if primary_role in ("PRIMARY_CREATOR", "SECONDARY_CREATOR", "SPOT_UP_WING"):
            return "G"
        return "F"

    ast = float(ast_pct or 0.0)
    trb = float(trb_pct or 0.0)
    blk = float(blk_pct or 0.0)
    tpar = float(three_par or 0.0)

    # Guards / wings who don't rebound like bigs.
    if (ast >= 18 and trb < 9.5) or (tpar >= 0.38 and trb < 9 and blk < 2.5) or (trb < 7 and blk < 1.5):
        return "G"
    # Centers / rebound-first bigs.
    if trb >= 15 or blk >= 3.5:
        return "C"
    # Forwards (including many PF/C tweeners that aren't pure G).
    if trb >= 9.5 or blk >= 1.5:
        return "F"
    if ast >= 12 or tpar >= 0.25:
        return "G"
    if primary_role in ("ROLL_BIG", "STRETCH_BIG"):
        return "C"
    return "F"


# Default starting five shape shown in the UI.
LINEUP_SLOTS = ("G", "G", "F", "F", "C")
