"""
web/app.py

Minimal Flask front end for game/engine.py's GameEngine.

Round state is returned as an HMAC-signed `ticket` so scoring works across
serverless instances (Vercel) that do not share process memory. The in-memory
pending map is kept as a fast path for single-process local dev.

Two play modes, both scored identically — the only difference is whether
PlayerCard.fame_tier/fame_score are included in the /api/round response:
  - competitive: hidden, guess from name/season/team alone
  - casual: shown, as a hint

Daily rounds use a UTC date seed so every player worldwide gets the same five
rosters. Daily always deals medium / competitive for a fair shared board.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, render_template, request

from game.engine import GameEngine, N_GAMES, TeamResult
from model.simulate import SeasonDistribution

N_SIMS = int(os.environ.get("GUESSTHEWINS_N_SIMS", "150"))
ROUND_SECRET = os.environ.get(
    "GUESSTHEWINS_ROUND_SECRET",
    "dev-only-change-me-in-production",
).encode("utf-8")

DAILY_DIFFICULTY = "medium"
DAILY_REVEAL_MODE = "competitive"  # no fame hints — fair shared board

# Slim runtime DB shipped for Vercel (full nba_data.db is too large to upload).
RUNTIME_DB_GZ = Path(__file__).resolve().parent / "nba_data_runtime.db.gz"
RUNTIME_DB_TMP = Path(os.environ.get("GUESSTHEWINS_DB_PATH", "/tmp/nba_data_runtime.db"))

app = Flask(__name__)


def _ensure_runtime_db() -> Path:
    """
    On Vercel, materialize the gzipped slim DB into /tmp once per instance.
    Locally, prefer the full repo DB when present.
    """
    full = Path(__file__).resolve().parents[1] / "nba_data.db"
    if full.exists() and not os.environ.get("VERCEL"):
        os.environ.setdefault("GUESSTHEWINS_DB_PATH", str(full))
        return full

    target = RUNTIME_DB_TMP
    if target.exists() and target.stat().st_size > 1_000_000:
        os.environ["GUESSTHEWINS_DB_PATH"] = str(target)
        return target

    if not RUNTIME_DB_GZ.exists():
        if full.exists():
            os.environ["GUESSTHEWINS_DB_PATH"] = str(full)
            return full
        raise FileNotFoundError("No nba_data.db or nba_data_runtime.db.gz found")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial")
    with gzip.open(RUNTIME_DB_GZ, "rb") as src, open(tmp, "wb") as dst:
        dst.write(src.read())
    tmp.replace(target)
    os.environ["GUESSTHEWINS_DB_PATH"] = str(target)
    return target


@lru_cache(maxsize=1)
def get_engine() -> GameEngine:
    """Lazy singleton — avoids loading models during Vercel's import probe."""
    # Import db after path env is set so modules that captured DB_PATH still
    # work via get_connection() which re-reads env... Engine takes db_path arg.
    db_path = _ensure_runtime_db()
    # Refresh db.DB_PATH for any code that reads the module global later.
    import db as db_mod

    db_mod.DB_PATH = db_path
    return GameEngine(db_path=db_path, n_sims=N_SIMS)


def _player_dict(player, mode: str) -> dict:
    d = {
        "player_id": player.player_id,
        "full_name": player.full_name,
        "season": player.season,
        "team_abbr": player.team_abbr,
        "position": player.position,
    }
    if mode == "casual":
        d["fame_tier"] = player.fame_tier
    return d


def _dist_payload(dist: SeasonDistribution) -> dict[str, Any]:
    return {
        "point_estimate_wins": dist.point_estimate_wins,
        "point_estimate_net_rating": dist.point_estimate_net_rating,
        "model_residual_std": dist.model_residual_std,
        "mean": dist.mean,
        "median": dist.median,
        "std": dist.std,
        "p10": dist.p10,
        "p25": dist.p25,
        "p75": dist.p75,
        "p90": dist.p90,
        "min": dist.min,
        "max": dist.max,
        "n_sims": dist.n_sims,
        "gbm_net_rating": dist.gbm_net_rating,
        "ridge_net_rating": dist.ridge_net_rating,
        "ood_score": dist.ood_score,
        "blend_weight": dist.blend_weight,
        "extrapolation_warning": dist.extrapolation_warning,
    }


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _daily_seed(date_str: str) -> int:
    """Stable int seed for a calendar day — same for every server/instance."""
    digest = hashlib.sha256(f"guessthewins-daily-v1-{date_str}".encode()).hexdigest()
    return int(digest[:8], 16)


def _mint_ticket(payload: dict) -> str:
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    sig = hmac.new(ROUND_SECRET, body.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _read_ticket(ticket: str) -> dict:
    try:
        body, sig = ticket.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError("invalid ticket") from exc
    expected = hmac.new(ROUND_SECRET, body.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise ValueError("invalid ticket signature")
    return json.loads(base64.urlsafe_b64decode(body.encode("ascii")))


def _score_from_ticket(ticket: str, team_index: int, guess: float) -> TeamResult:
    if not 0.0 <= guess <= 1.0:
        raise ValueError(f"Guess for team {team_index} must be a win rate in [0, 1], got {guess}")

    payload = _read_ticket(ticket)
    teams = payload.get("teams") or {}
    key = str(team_index)
    if key not in teams:
        raise ValueError(f"Unknown team_index {team_index!r} for ticket")

    entry = teams[key]
    dist = SeasonDistribution(**entry["distribution"])
    actual_win_pct = dist.point_estimate_wins / N_GAMES
    error = abs(guess - actual_win_pct)

    # Reconstruct a minimal PlayerCard-like list for the JSON response.
    from game.roster import PlayerCard

    players = [
        PlayerCard(
            player_id=p["player_id"],
            full_name=p["full_name"],
            season=p["season"],
            team_id=p.get("team_id", 0),
            team_abbr=p["team_abbr"],
            fame_tier=p.get("fame_tier", "role_player"),
            fame_score=float(p.get("fame_score", 0.0)),
            position=p.get("position", "F"),
        )
        for p in entry["players"]
    ]

    return TeamResult(
        team_index=team_index,
        players=players,
        guess_win_pct=guess,
        actual_win_pct=actual_win_pct,
        actual_wins=dist.point_estimate_wins,
        error=error,
        distribution=dist,
    )


def _team_result_json(r: TeamResult) -> dict:
    return {
        "team_index": r.team_index,
        "players": [
            {
                "player_id": p.player_id,
                "full_name": p.full_name,
                "season": p.season,
                "team_abbr": p.team_abbr,
                "fame_tier": p.fame_tier,
            }
            for p in r.players
        ],
        "guess_win_pct": r.guess_win_pct,
        "actual_win_pct": r.actual_win_pct,
        "actual_wins": r.actual_wins,
        "error": r.error,
        "p10_wins": r.distribution.p10,
        "p90_wins": r.distribution.p90,
        "extrapolation_warning": r.distribution.extrapolation_warning,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/daily")
def daily_info():
    """Lightweight endpoint so the client can show today's date without dealing."""
    date_str = _utc_today()
    return jsonify(
        {
            "daily_date": date_str,
            "difficulty": DAILY_DIFFICULTY,
            "mode": DAILY_REVEAL_MODE,
        }
    )


@app.route("/api/round", methods=["POST"])
def new_round():
    body = request.get_json(force=True, silent=True) or {}
    is_daily = bool(body.get("daily"))
    seed: Optional[int] = None
    daily_date: Optional[str] = None

    if is_daily:
        daily_date = _utc_today()
        difficulty = DAILY_DIFFICULTY
        mode = DAILY_REVEAL_MODE
        seed = _daily_seed(daily_date)
    else:
        difficulty = body.get("difficulty", "medium")
        mode = body.get("mode", "competitive")
        if difficulty not in ("easy", "medium", "hard"):
            return jsonify({"error": f"invalid difficulty {difficulty!r}"}), 400
        if mode not in ("competitive", "casual"):
            return jsonify({"error": f"invalid mode {mode!r}"}), 400

    engine = get_engine()
    try:
        rnd = engine.new_round(difficulty=difficulty, seed=seed)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    pending = engine._pending[rnd.round_id]
    ticket_teams = {}
    for team in rnd.teams:
        dist = pending.actuals[team.team_index]
        ticket_teams[str(team.team_index)] = {
            "players": [
                {
                    "player_id": p.player_id,
                    "full_name": p.full_name,
                    "season": p.season,
                    "team_id": p.team_id,
                    "team_abbr": p.team_abbr,
                    "position": p.position,
                    "fame_tier": p.fame_tier,
                    "fame_score": p.fame_score,
                }
                for p in team.players
            ],
            "distribution": _dist_payload(dist),
        }

    ticket_payload = {
        "round_id": rnd.round_id,
        "difficulty": rnd.difficulty,
        "teams": ticket_teams,
    }
    if daily_date:
        ticket_payload["daily_date"] = daily_date

    ticket = _mint_ticket(ticket_payload)

    response = {
        "round_id": rnd.round_id,
        "ticket": ticket,
        "difficulty": rnd.difficulty,
        "mode": mode,
        "daily": is_daily,
        "teams": [
            {
                "team_index": team.team_index,
                "players": [_player_dict(p, mode) for p in team.players],
            }
            for team in rnd.teams
        ],
    }
    if daily_date:
        response["daily_date"] = daily_date
    return jsonify(response)


@app.route("/api/score_team", methods=["POST"])
def score_team():
    body = request.get_json(force=True, silent=True) or {}
    round_id = body.get("round_id")
    ticket = body.get("ticket")
    raw_team_index = body.get("team_index")
    raw_guess = body.get("guess")

    if raw_team_index is None or raw_guess is None:
        return jsonify({"error": "team_index and guess are required"}), 400
    if not ticket and not round_id:
        return jsonify({"error": "ticket or round_id is required"}), 400

    try:
        team_index = int(raw_team_index)
        guess = float(raw_guess)
    except (TypeError, ValueError):
        return jsonify({"error": "team_index must be an int and guess must be a win rate"}), 400

    try:
        if ticket:
            r = _score_from_ticket(ticket, team_index, guess)
        else:
            r = get_engine().score_team(round_id, team_index, guess)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"round_id": round_id or "", **_team_result_json(r)})


@app.route("/api/score", methods=["POST"])
def score_round():
    body = request.get_json(force=True, silent=True) or {}
    round_id = body.get("round_id")
    ticket = body.get("ticket")
    raw_guesses = body.get("guesses")

    if not isinstance(raw_guesses, dict) or not raw_guesses:
        return jsonify({"error": "guesses are required"}), 400

    try:
        guesses = {int(k): float(v) for k, v in raw_guesses.items()}
    except (TypeError, ValueError):
        return jsonify({"error": "guesses must map team_index -> win rate"}), 400

    try:
        if ticket:
            payload = _read_ticket(ticket)
            team_results = [
                _score_from_ticket(ticket, idx, guesses[idx])
                for idx in sorted(guesses)
            ]
            total_error = sum(r.error for r in team_results)
            return jsonify(
                {
                    "round_id": payload.get("round_id", ""),
                    "difficulty": payload.get("difficulty", ""),
                    "total_error": total_error,
                    "average_error": total_error / len(team_results),
                    "team_results": [_team_result_json(r) for r in team_results],
                }
            )

        if not round_id:
            return jsonify({"error": "round_id or ticket is required"}), 400
        result = get_engine().score_round(round_id, guesses)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "round_id": result.round_id,
            "difficulty": result.difficulty,
            "total_error": result.total_error,
            "average_error": result.average_error,
            "team_results": [_team_result_json(r) for r in result.team_results],
        }
    )


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Warm the engine once for local play.
    get_engine()
    app.run(debug=True, port=5050)
