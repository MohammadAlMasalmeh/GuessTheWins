"""
game/engine.py

Round-based game loop on top of RosterGenerator (game/roster.py) and
WinPredictor (model/simulate.py):

  1. new_round() deals `n_teams` (default 5) fame-weighted fictional rosters,
     without revealing the model's win-rate estimate for any of them.
  2. The caller (CLI/web/API layer) collects the player's win% guess for
     each team.
  3. score_round() compares guesses to the model's point-estimate win% and
     reports each team's absolute error plus the round's total error. 0 is a
     perfect round; lower is better.

No UI here — this module only holds game state and scoring rules.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import db
from game.roster import PlayerCard, RosterGenerator
from model.simulate import SeasonDistribution, WinPredictor

DEFAULT_N_TEAMS = 5
N_GAMES = 82


@dataclass(frozen=True)
class TeamCard:
    team_index: int
    players: List[PlayerCard]


@dataclass(frozen=True)
class Round:
    round_id: str
    difficulty: str
    teams: List[TeamCard]


@dataclass(frozen=True)
class TeamResult:
    team_index: int
    players: List[PlayerCard]
    guess_win_pct: float
    actual_win_pct: float
    actual_wins: float
    error: float
    distribution: SeasonDistribution


@dataclass(frozen=True)
class RoundResult:
    round_id: str
    difficulty: str
    team_results: List[TeamResult]
    total_error: float
    average_error: float


@dataclass
class _PendingRound:
    difficulty: str
    teams: List[TeamCard]
    actuals: Dict[int, SeasonDistribution]
    scored: set = field(default_factory=set)


class GameEngine:
    def __init__(self, db_path=db.DB_PATH, n_sims: int = 2000):
        self.predictor = WinPredictor(db_path=db_path)
        self.roster_gen = RosterGenerator(db_path=db_path)
        self.n_sims = n_sims
        self._pending: Dict[str, _PendingRound] = {}

    def new_round(
        self,
        difficulty: str = "medium",
        n_teams: int = DEFAULT_N_TEAMS,
        seed: Optional[int] = None,
    ) -> Round:
        """Deal `n_teams` rosters. No player across the round's teams repeats."""
        teams: List[TeamCard] = []
        actuals: Dict[int, SeasonDistribution] = {}
        used_player_ids: set = set()

        for i in range(n_teams):
            team_seed = None if seed is None else seed * 1000 + i
            players = self.roster_gen.sample_roster(
                difficulty=difficulty, seed=team_seed, exclude_player_ids=used_player_ids
            )
            used_player_ids.update(p.player_id for p in players)

            player_seasons = [(p.player_id, p.season, p.team_id) for p in players]
            dist = self.predictor.simulate_by_player_ids(
                player_seasons, n_sims=self.n_sims, seed=team_seed
            )

            teams.append(TeamCard(team_index=i, players=players))
            actuals[i] = dist

        round_id = str(uuid.uuid4())
        self._pending[round_id] = _PendingRound(difficulty=difficulty, teams=teams, actuals=actuals)
        return Round(round_id=round_id, difficulty=difficulty, teams=teams)

    def score_team(self, round_id: str, team_index: int, guess: float) -> TeamResult:
        """
        Score a single team within an in-progress round without popping the
        round, so the remaining teams can still be scored one at a time.
        Once every dealt team has been scored, the round is popped.
        """
        pending = self._pending.get(round_id)
        if pending is None:
            raise ValueError(f"Unknown or already-scored round_id: {round_id!r}")

        if team_index not in pending.actuals:
            raise ValueError(f"Unknown team_index {team_index!r} for round_id {round_id!r}")
        if team_index in pending.scored:
            raise ValueError("team already scored")
        if not 0.0 <= guess <= 1.0:
            raise ValueError(f"Guess for team {team_index} must be a win rate in [0, 1], got {guess}")

        dist = pending.actuals[team_index]
        actual_win_pct = dist.point_estimate_wins / N_GAMES
        error = abs(guess - actual_win_pct)

        result = TeamResult(
            team_index=team_index,
            players=pending.teams[team_index].players,
            guess_win_pct=guess,
            actual_win_pct=actual_win_pct,
            actual_wins=dist.point_estimate_wins,
            error=error,
            distribution=dist,
        )

        pending.scored.add(team_index)
        if pending.scored >= set(pending.actuals):
            self._pending.pop(round_id, None)

        return result

    def score_round(self, round_id: str, guesses: Dict[int, float]) -> RoundResult:
        """
        `guesses`: {team_index: guessed win% in [0, 1]}, one entry per team
        dealt in the round. Scoring consumes the round — call once per round.
        """
        pending = self._pending.pop(round_id, None)
        if pending is None:
            raise ValueError(f"Unknown or already-scored round_id: {round_id!r}")

        expected = set(pending.actuals)
        if set(guesses) != expected:
            raise ValueError(f"Must supply exactly one guess per team index {sorted(expected)}")

        team_results = []
        for i in sorted(expected):
            guess = guesses[i]
            if not 0.0 <= guess <= 1.0:
                raise ValueError(f"Guess for team {i} must be a win rate in [0, 1], got {guess}")
            dist = pending.actuals[i]
            actual_win_pct = dist.point_estimate_wins / N_GAMES
            error = abs(guess - actual_win_pct)
            team_results.append(
                TeamResult(
                    team_index=i,
                    players=pending.teams[i].players,
                    guess_win_pct=guess,
                    actual_win_pct=actual_win_pct,
                    actual_wins=dist.point_estimate_wins,
                    error=error,
                    distribution=dist,
                )
            )

        total_error = sum(r.error for r in team_results)
        return RoundResult(
            round_id=round_id,
            difficulty=pending.difficulty,
            team_results=team_results,
            total_error=total_error,
            average_error=total_error / len(team_results),
        )
