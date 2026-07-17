"""
game/roster.py

Fame-weighted player-season sampling for building game rosters.

The DB has ~5,400 players spanning 1946-2026; sampling uniformly at random
would fill most rosters with players nobody has heard of, which defeats the
point of a "guess the win%" game (you can't reason about a team of
strangers). Instead, sample players weighted by `player_fame_tier.tier`
(LEGEND/STAR/SOLID_STARTER/ROLE_PLAYER/OBSCURE, computed in
compute_fame_tier.py from career points/win-shares/VORP/accolades) so
recognizable players dominate, with a `difficulty` knob controlling how much
long-tail/obscure talent is allowed to mix in.

Rosters are dealt into fixed lineup slots (G, G, F, F, C) so a Guard slot
never gets Tim Duncan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

import db
from game.positions import LINEUP_SLOTS, classify_position

# A player-season must clear this bar to be sampled at all, so nobody gets
# handed a 4-game September call-up as their "known" season.
MIN_SEASON_MINUTES = 500.0
MIN_SEASON_GAMES = 20

TIERS = ["LEGEND", "STAR", "SOLID_STARTER", "ROLE_PLAYER", "OBSCURE"]

# Total sampling probability mass allocated to each fame tier, split evenly
# across the players *in that tier's eligible pool* (see
# RosterGenerator._sampling_weights) — so e.g. "easy" massively overweights
# LEGEND despite it being the smallest tier. Values are relative, not
# required to sum to anything; they're renormalized per roster.
DIFFICULTY_TIER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "easy":   {"LEGEND": 55, "STAR": 30, "SOLID_STARTER": 10, "ROLE_PLAYER": 4,  "OBSCURE": 1},
    "medium": {"LEGEND": 30, "STAR": 28, "SOLID_STARTER": 22, "ROLE_PLAYER": 13, "OBSCURE": 7},
    "hard":   {"LEGEND": 14, "STAR": 16, "SOLID_STARTER": 20, "ROLE_PLAYER": 22, "OBSCURE": 28},
}


@dataclass(frozen=True)
class PlayerCard:
    player_id: int
    full_name: str
    season: str
    team_id: int
    team_abbr: Optional[str]
    fame_tier: str
    fame_score: float
    position: str  # "G" | "F" | "C"


@dataclass(frozen=True)
class _SeasonOption:
    season: str
    team_id: int
    weight: float  # minutes_total that season
    ast_pct: Optional[float]
    trb_pct: Optional[float]
    blk_pct: Optional[float]
    three_par: Optional[float]


class RosterGenerator:
    """Loads the fame/season pools once, then samples rosters cheaply."""

    def __init__(self, db_path=db.DB_PATH):
        self.db_path = db_path
        self._players: Dict[int, dict] = {}
        self._seasons: Dict[int, List[_SeasonOption]] = {}
        self._tier_player_ids: Dict[str, List[int]] = {t: [] for t in TIERS}
        self._position_player_ids: Dict[str, List[int]] = {"G": [], "F": [], "C": []}
        self._team_abbr: Dict[int, str] = {}
        self._roles: Dict[tuple, str] = {}
        self._load()

    def _load(self) -> None:
        conn = db.get_connection(self.db_path)
        try:
            for team_id, abbr in conn.execute("SELECT team_id, abbreviation FROM teams"):
                self._team_abbr[team_id] = abbr

            for player_id, season, role in conn.execute(
                "SELECT player_id, season, primary_role FROM player_roles"
            ):
                self._roles[(player_id, season)] = role

            for row in conn.execute(
                """
                SELECT player_id, season, team_id, minutes_total,
                       ast_pct, trb_pct, blk_pct, three_par
                FROM player_seasons
                WHERE minutes_total IS NOT NULL AND minutes_total >= ?
                  AND games_played IS NOT NULL AND games_played >= ?
                """,
                (MIN_SEASON_MINUTES, MIN_SEASON_GAMES),
            ).fetchall():
                player_id, season, team_id, minutes_total, ast_pct, trb_pct, blk_pct, three_par = row
                self._seasons.setdefault(player_id, []).append(
                    _SeasonOption(
                        season=season,
                        team_id=team_id,
                        weight=minutes_total,
                        ast_pct=ast_pct,
                        trb_pct=trb_pct,
                        blk_pct=blk_pct,
                        three_par=three_par,
                    )
                )

            for player_id, full_name, tier, fame_score in conn.execute(
                """
                SELECT p.player_id, p.full_name, f.tier, f.fame_score
                FROM players p
                JOIN player_fame_tier f ON f.player_id = p.player_id
                """
            ).fetchall():
                if player_id not in self._seasons:
                    continue  # no season clears the minutes/games floor
                # Position from the player's highest-minutes qualifying season.
                best = max(self._seasons[player_id], key=lambda o: o.weight)
                role = self._roles.get((player_id, best.season))
                position = classify_position(
                    best.ast_pct, best.trb_pct, best.blk_pct, best.three_par, role
                )
                self._players[player_id] = {
                    "full_name": full_name,
                    "tier": tier,
                    "fame_score": fame_score,
                    "position": position,
                }
                self._tier_player_ids.setdefault(tier, []).append(player_id)
                self._position_player_ids.setdefault(position, []).append(player_id)
        finally:
            conn.close()

    def _sampling_weights(self, difficulty: str) -> Dict[int, float]:
        tier_weights = DIFFICULTY_TIER_WEIGHTS.get(difficulty)
        if tier_weights is None:
            raise ValueError(f"Unknown difficulty {difficulty!r}; choose from {list(DIFFICULTY_TIER_WEIGHTS)}")
        weights: Dict[int, float] = {}
        for tier, mass in tier_weights.items():
            pool = self._tier_player_ids.get(tier, [])
            if not pool or mass <= 0:
                continue
            per_player = mass / len(pool)
            for pid in pool:
                weights[pid] = per_player
        return weights

    def _pick_season(self, player_id: int, rng: np.random.Generator) -> _SeasonOption:
        options = self._seasons[player_id]
        if len(options) == 1:
            return options[0]
        w = np.array([o.weight for o in options], dtype=float)
        w = w / w.sum()
        idx = rng.choice(len(options), p=w)
        return options[idx]

    def _pick_for_slot(
        self,
        slot: str,
        weights: Dict[int, float],
        exclude: Set[int],
        rng: np.random.Generator,
    ) -> int:
        """Pick one player whose inferred position matches `slot`."""
        pool = [
            pid for pid in self._position_player_ids.get(slot, [])
            if pid in weights and pid not in exclude
        ]
        # Rare fallback if a position pool is exhausted mid-round: widen to
        # neighboring big buckets (F↔C) before failing — never put a C in G.
        if not pool and slot == "C":
            pool = [
                pid for pid in self._position_player_ids.get("F", [])
                if pid in weights and pid not in exclude
            ]
        if not pool and slot == "F":
            pool = [
                pid for pid in self._position_player_ids.get("C", [])
                if pid in weights and pid not in exclude
            ]
        if not pool:
            raise ValueError(f"Not enough eligible {slot} players to fill lineup slot")

        probs = np.array([weights[pid] for pid in pool], dtype=float)
        probs = probs / probs.sum()
        return int(pool[rng.choice(len(pool), p=probs)])

    def sample_roster(
        self,
        n: int = 5,
        difficulty: str = "medium",
        seed: Optional[int] = None,
        exclude_player_ids: Optional[Set[int]] = None,
        slots: Optional[tuple] = None,
    ) -> List[PlayerCard]:
        """
        Sample a 5-man roster into lineup slots (default G/G/F/F/C), fame-
        weighted per `difficulty`. Each player is paired with one of their
        qualifying seasons (weighted by minutes).
        """
        if n != 5 and slots is None:
            raise ValueError("Non-5 rosters require an explicit slots tuple")
        lineup_slots = slots or LINEUP_SLOTS
        if len(lineup_slots) != n:
            raise ValueError(f"slots length {len(lineup_slots)} != n={n}")

        rng = np.random.default_rng(seed)
        weights = self._sampling_weights(difficulty)
        exclude = set(exclude_player_ids or set())

        cards: List[PlayerCard] = []
        for slot in lineup_slots:
            player_id = self._pick_for_slot(slot, weights, exclude, rng)
            exclude.add(player_id)
            info = self._players[player_id]
            season_opt = self._pick_season(player_id, rng)
            cards.append(
                PlayerCard(
                    player_id=player_id,
                    full_name=info["full_name"],
                    season=season_opt.season,
                    team_id=season_opt.team_id,
                    team_abbr=self._team_abbr.get(season_opt.team_id),
                    fame_tier=info["tier"],
                    fame_score=info["fame_score"],
                    position=slot,
                )
            )
        return cards
