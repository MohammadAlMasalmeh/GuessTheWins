"""
db.py

Schema and connection handling for the NBA win-prediction game database.

Design goals:
- Capture essentially every commonly-tracked player and team advanced stat,
  since we don't yet know which features the win-prediction model will need.
  Over-collecting now is cheap; re-scraping later is not.
- Normalized structure with clear foreign keys, portable to Postgres later
  (standard types only, no SQLite-only features).
- Every table keyed so upserts are trivial (safe to re-run scrapers).

This file only defines schema and connection helpers. Scraping scripts
(scrape_players.py, scrape_teams.py, scrape_lineups.py) populate it.
"""

import os
import sqlite3
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent / "nba_data.db"
DB_PATH = Path(os.environ.get("GUESSTHEWINS_DB_PATH", str(_DEFAULT_DB)))

SCHEMA = """
-- ============================================================
-- CORE ENTITIES
-- ============================================================

CREATE TABLE IF NOT EXISTS players (
    player_id       INTEGER PRIMARY KEY,
    full_name       TEXT NOT NULL,
    height_inches   INTEGER,
    weight_lbs      INTEGER,
    primary_position TEXT,
    draft_year      INTEGER,
    draft_round     INTEGER,
    draft_pick      INTEGER,
    college_or_country TEXT,
    career_start_season TEXT,
    career_end_season   TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id         INTEGER PRIMARY KEY,
    full_name       TEXT NOT NULL,
    abbreviation    TEXT NOT NULL,
    city            TEXT,
    conference      TEXT,
    division        TEXT,
    active_from     TEXT,
    active_to       TEXT
);

-- ============================================================
-- PLAYER-SEASON: every commonly tracked box score + advanced stat
-- ============================================================

CREATE TABLE IF NOT EXISTS player_seasons (
    player_id       INTEGER NOT NULL,
    season          TEXT NOT NULL,          -- e.g. "2015-16"
    team_id         INTEGER NOT NULL,       -- if traded mid-season, one row per team stint
    age             INTEGER,

    -- Playing time
    games_played    INTEGER,
    games_started   INTEGER,
    minutes_total   REAL,
    minutes_per_game REAL,

    -- Traditional box score totals
    pts             REAL,
    fgm             REAL,
    fga             REAL,
    fg3m            REAL,
    fg3a            REAL,
    ftm             REAL,
    fta             REAL,
    oreb            REAL,
    dreb            REAL,
    reb             REAL,
    ast             REAL,
    stl             REAL,
    blk             REAL,
    tov             REAL,
    pf              REAL,

    -- Traditional box score per-game (derived from totals / games_played;
    -- stored for easy game/query access — always recompute if GP or totals change)
    pts_per_game    REAL,
    fgm_per_game    REAL,
    fga_per_game    REAL,
    fg3m_per_game   REAL,
    fg3a_per_game   REAL,
    ftm_per_game    REAL,
    fta_per_game    REAL,
    oreb_per_game   REAL,
    dreb_per_game   REAL,
    reb_per_game    REAL,
    ast_per_game    REAL,
    stl_per_game    REAL,
    blk_per_game    REAL,
    tov_per_game    REAL,
    pf_per_game     REAL,

    -- Shooting efficiency
    fg_pct          REAL,
    fg3_pct         REAL,
    ft_pct          REAL,
    efg_pct         REAL,           -- effective FG%
    ts_pct          REAL,           -- true shooting %

    -- Shot profile / creation (era-gated: 3PAr from 1979-80, AST/TO from 1977-78)
    three_par       REAL,           -- 3PA / FGA
    ft_rate         REAL,           -- FTA / FGA
    ast_to          REAL,           -- AST / TOV

    -- Advanced rate stats
    per             REAL,           -- player efficiency rating
    usg_pct         REAL,           -- usage rate
    ast_pct         REAL,
    orb_pct         REAL,
    drb_pct         REAL,
    trb_pct         REAL,
    stl_pct         REAL,
    blk_pct         REAL,
    tov_pct         REAL,
    pie             REAL,           -- player impact estimate

    -- Win shares
    ows             REAL,           -- offensive win shares
    dws             REAL,           -- defensive win shares
    ws              REAL,           -- total win shares
    ws_per_48       REAL,

    -- Plus-minus family
    obpm            REAL,           -- offensive box plus-minus
    dbpm            REAL,           -- defensive box plus-minus
    bpm             REAL,           -- total box plus-minus
    vorp            REAL,           -- value over replacement player

    -- On-court team performance while this player plays (context stats)
    on_court_off_rating   REAL,
    on_court_def_rating   REAL,
    on_court_net_rating   REAL,
    off_court_net_rating  REAL,     -- team's net rating when player is OFF the court
    net_rating_on_off_diff REAL,    -- on_court_net_rating - off_court_net_rating

    -- Clutch performance (optional, best-effort where available)
    clutch_pts_per_game    REAL,
    clutch_plus_minus      REAL,

    -- Data provenance: many stats simply do not exist for older seasons
    -- (e.g. no 3-pointers before 1979-80, no steals/blocks/turnovers before
    -- 1973-74, no on/off or lineup data before roughly 2007-08). Leave the
    -- relevant columns above NULL rather than 0 in those cases, since 0 would
    -- silently look like real data. Use these two columns to record whether
    -- a stat that DOES exist for the row was official league tracking or a
    -- reconstructed estimate (e.g. Basketball-Reference's regression-based
    -- estimates of steals/blocks/BPM for some pre-1974 players like Wilt
    -- Chamberlain). We still store estimates, just flagged as such.
    has_estimated_stats    INTEGER DEFAULT 0,   -- 0/1: were any fields above filled via estimation rather than official tracking
    estimate_source_note   TEXT,                -- free text, e.g. "blocks/BPM estimated by Basketball-Reference regression model"

    PRIMARY KEY (player_id, season, team_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- ============================================================
-- ACCOLADES: separate table since a player-season can have several
-- ============================================================

CREATE TABLE IF NOT EXISTS player_accolades (
    player_id       INTEGER NOT NULL,
    season          TEXT NOT NULL,
    accolade_type   TEXT NOT NULL,   -- e.g. "ALL_STAR", "ALL_NBA", "ALL_DEFENSIVE", "MVP_VOTE", "DPOY_VOTE", "ROY", "SIXTH_MAN"
    tier_or_rank    TEXT,            -- e.g. "1st Team", "2nd Team", numeric MVP vote rank, etc.

    PRIMARY KEY (player_id, season, accolade_type),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- ============================================================
-- TEAM-SEASON: outcome (label) + full advanced context (features)
-- ============================================================

CREATE TABLE IF NOT EXISTS team_seasons (
    team_id         INTEGER NOT NULL,
    season          TEXT NOT NULL,

    -- Outcome (this is the label the model predicts)
    wins            INTEGER,
    losses          INTEGER,
    win_pct         REAL,

    -- Scoring
    pts_per_game        REAL,
    opp_pts_per_game    REAL,
    point_diff_per_game REAL,

    -- Core advanced ratings
    off_rating      REAL,
    def_rating      REAL,
    net_rating      REAL,
    pace            REAL,

    -- Four factors (team)
    team_efg_pct    REAL,
    team_tov_pct    REAL,
    team_orb_pct    REAL,
    team_ft_rate    REAL,           -- FTA / FGA

    -- Four factors (opponent, i.e. defense against these)
    opp_efg_pct     REAL,
    opp_tov_pct     REAL,
    opp_orb_pct     REAL,
    opp_ft_rate     REAL,

    -- Strength-of-schedule adjusted metrics
    srs             REAL,           -- simple rating system (Basketball-Reference)
    sos             REAL,           -- strength of schedule
    mov             REAL,           -- margin of victory

    -- Playoffs
    made_playoffs   INTEGER,        -- 0/1
    playoff_seed    INTEGER,
    playoff_result  TEXT,           -- e.g. "LOST_FIRST_ROUND", "WON_FINALS"

    -- Same provenance logic as player_seasons
    has_estimated_stats    INTEGER DEFAULT 0,
    estimate_source_note   TEXT,

    PRIMARY KEY (team_id, season),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- ============================================================
-- LINEUPS: 5-man combinations (best-effort, ~2007-08 onward)
-- ============================================================

CREATE TABLE IF NOT EXISTS lineups (
    lineup_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    season          TEXT NOT NULL,
    team_id         INTEGER NOT NULL,
    player_id_1     INTEGER NOT NULL,
    player_id_2     INTEGER NOT NULL,
    player_id_3     INTEGER NOT NULL,
    player_id_4     INTEGER NOT NULL,
    player_id_5     INTEGER NOT NULL,

    minutes_played  REAL,
    off_rating      REAL,
    def_rating      REAL,
    net_rating      REAL,
    pace            REAL,

    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (player_id_1) REFERENCES players(player_id),
    FOREIGN KEY (player_id_2) REFERENCES players(player_id),
    FOREIGN KEY (player_id_3) REFERENCES players(player_id),
    FOREIGN KEY (player_id_4) REFERENCES players(player_id),
    FOREIGN KEY (player_id_5) REFERENCES players(player_id),

    -- prevent exact duplicate lineup rows from repeated scrapes
    UNIQUE (season, team_id, player_id_1, player_id_2, player_id_3, player_id_4, player_id_5)
);

-- ============================================================
-- DERIVED: recognizability tier for game design (computed, not scraped)
-- ============================================================

CREATE TABLE IF NOT EXISTS player_fame_tier (
    player_id       INTEGER PRIMARY KEY,
    tier            TEXT NOT NULL,      -- LEGEND / STAR / SOLID_STARTER / ROLE_PLAYER / OBSCURE
    fame_score      REAL,               -- raw computed score, kept for re-tuning tier cutoffs later

    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- ============================================================
-- STAT AVAILABILITY: documents which stat categories exist at all for a
-- given season, so scrapers and feature-engineering code can check
-- "should this be populated?" instead of assuming, and instead of
-- guessing based on a NULL alone (NULL can also mean "scrape failed").
-- ============================================================

CREATE TABLE IF NOT EXISTS stat_availability (
    season          TEXT NOT NULL,
    stat_category   TEXT NOT NULL,   -- e.g. "THREE_POINT", "STEALS_BLOCKS", "TURNOVERS", "ADVANCED_BPM_VORP", "ON_OFF_LINEUPS", "PLAY_BY_PLAY_DERIVED"
    available       INTEGER NOT NULL,  -- 0/1
    note            TEXT,

    PRIMARY KEY (season, stat_category)
);

-- Helpful indexes for the aggregation queries the modeling phase will run
CREATE INDEX IF NOT EXISTS idx_player_seasons_season ON player_seasons(season);
CREATE INDEX IF NOT EXISTS idx_player_seasons_team_season ON player_seasons(team_id, season);
CREATE INDEX IF NOT EXISTS idx_team_seasons_season ON team_seasons(season);
CREATE INDEX IF NOT EXISTS idx_lineups_season_team ON lineups(season, team_id);

-- ============================================================
-- LINEUP COMBOS: 2-man / 3-man / 5-man (nba_api; ~2007-08+)
-- Separate from `lineups` so 2/3-man rows can leave trailing player_ids NULL.
-- ============================================================

CREATE TABLE IF NOT EXISTS lineup_combos (
    combo_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    season          TEXT NOT NULL,
    team_id         INTEGER NOT NULL,
    combo_size      INTEGER NOT NULL,   -- 2, 3, or 5
    player_id_1     INTEGER NOT NULL,
    player_id_2     INTEGER NOT NULL,
    player_id_3     INTEGER,
    player_id_4     INTEGER,
    player_id_5     INTEGER,
    minutes_played  REAL,
    off_rating      REAL,
    def_rating      REAL,
    net_rating      REAL,
    pace            REAL,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (player_id_1) REFERENCES players(player_id),
    FOREIGN KEY (player_id_2) REFERENCES players(player_id),
    FOREIGN KEY (player_id_3) REFERENCES players(player_id),
    FOREIGN KEY (player_id_4) REFERENCES players(player_id),
    FOREIGN KEY (player_id_5) REFERENCES players(player_id),
    UNIQUE (season, team_id, combo_size, player_id_1, player_id_2, player_id_3, player_id_4, player_id_5)
);

CREATE INDEX IF NOT EXISTS idx_lineup_combos_season_team ON lineup_combos(season, team_id, combo_size);

-- ============================================================
-- PUBLIC IMPACT METRICS: DARKO / RAPM / LEBRON / EPM / box proxies
-- One row per player-season-source. Era coverage varies by source.
-- ============================================================

CREATE TABLE IF NOT EXISTS player_impact_metrics (
    player_id       INTEGER NOT NULL,
    season          TEXT NOT NULL,
    metric_source   TEXT NOT NULL,   -- DARKO_DPM, BOX_DPM, RAPM_1Y, RAPM_3Y, LEBRON, EPM, BOX_BPM_PROXY
    offense         REAL,
    defense         REAL,
    total           REAL,
    wins_added      REAL,
    sample_note     TEXT,            -- e.g. partial historical reconstruction
    PRIMARY KEY (player_id, season, metric_source),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE INDEX IF NOT EXISTS idx_impact_season ON player_impact_metrics(season, metric_source);

-- ============================================================
-- PLAYER TRACKING (SportVU / Second Spectrum style; ~2013-14+)
-- ============================================================

CREATE TABLE IF NOT EXISTS player_tracking_seasons (
    player_id       INTEGER NOT NULL,
    season          TEXT NOT NULL,
    team_id         INTEGER NOT NULL,
    dist_miles      REAL,
    avg_speed       REAL,
    drives          REAL,
    drive_pts       REAL,
    passes_made     REAL,
    secondary_ast   REAL,
    potential_ast   REAL,
    ast_points_created REAL,
    oreb_contest    REAL,
    dreb_contest    REAL,
    PRIMARY KEY (player_id, season, team_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- ============================================================
-- PLAYER ROLES (derived heuristics from box + shot profile)
-- ============================================================

CREATE TABLE IF NOT EXISTS player_roles (
    player_id       INTEGER NOT NULL,
    season          TEXT NOT NULL,
    primary_role    TEXT NOT NULL,   -- e.g. PRIMARY_CREATOR, SECONDARY_CREATOR, SPOT_UP_WING, ROLL_BIG, STRETCH_BIG, DEFENSIVE_SPECIALIST, BENCH_SCORER
    role_confidence REAL,
    position_versatility REAL,      -- crude 0-1 from multi-position / usage mix
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- Easy query surface: same rows as player_seasons with per-game columns
-- guaranteed (view recomputes from totals when GP > 0). Prefer reading
-- pts_per_game etc. from player_seasons directly once scrapers have filled them;
-- this view is a safety net / convenience alias.
CREATE VIEW IF NOT EXISTS v_player_season_per_game AS
SELECT
    player_id,
    season,
    team_id,
    age,
    games_played,
    games_started,
    minutes_total,
    minutes_per_game,
    pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf,
    CASE WHEN games_played > 0 AND pts  IS NOT NULL THEN ROUND(pts  * 1.0 / games_played, 2) END AS pts_per_game,
    CASE WHEN games_played > 0 AND fgm  IS NOT NULL THEN ROUND(fgm  * 1.0 / games_played, 2) END AS fgm_per_game,
    CASE WHEN games_played > 0 AND fga  IS NOT NULL THEN ROUND(fga  * 1.0 / games_played, 2) END AS fga_per_game,
    CASE WHEN games_played > 0 AND fg3m IS NOT NULL THEN ROUND(fg3m * 1.0 / games_played, 2) END AS fg3m_per_game,
    CASE WHEN games_played > 0 AND fg3a IS NOT NULL THEN ROUND(fg3a * 1.0 / games_played, 2) END AS fg3a_per_game,
    CASE WHEN games_played > 0 AND ftm  IS NOT NULL THEN ROUND(ftm  * 1.0 / games_played, 2) END AS ftm_per_game,
    CASE WHEN games_played > 0 AND fta  IS NOT NULL THEN ROUND(fta  * 1.0 / games_played, 2) END AS fta_per_game,
    CASE WHEN games_played > 0 AND oreb IS NOT NULL THEN ROUND(oreb * 1.0 / games_played, 2) END AS oreb_per_game,
    CASE WHEN games_played > 0 AND dreb IS NOT NULL THEN ROUND(dreb * 1.0 / games_played, 2) END AS dreb_per_game,
    CASE WHEN games_played > 0 AND reb  IS NOT NULL THEN ROUND(reb  * 1.0 / games_played, 2) END AS reb_per_game,
    CASE WHEN games_played > 0 AND ast  IS NOT NULL THEN ROUND(ast  * 1.0 / games_played, 2) END AS ast_per_game,
    CASE WHEN games_played > 0 AND stl  IS NOT NULL THEN ROUND(stl  * 1.0 / games_played, 2) END AS stl_per_game,
    CASE WHEN games_played > 0 AND blk  IS NOT NULL THEN ROUND(blk  * 1.0 / games_played, 2) END AS blk_per_game,
    CASE WHEN games_played > 0 AND tov  IS NOT NULL THEN ROUND(tov  * 1.0 / games_played, 2) END AS tov_per_game,
    CASE WHEN games_played > 0 AND pf   IS NOT NULL THEN ROUND(pf   * 1.0 / games_played, 2) END AS pf_per_game,
    fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct,
    per, usg_pct, ast_pct, orb_pct, drb_pct, trb_pct, stl_pct, blk_pct, tov_pct, pie,
    ows, dws, ws, ws_per_48,
    obpm, dbpm, bpm, vorp,
    on_court_off_rating, on_court_def_rating, on_court_net_rating,
    off_court_net_rating, net_rating_on_off_diff,
    clutch_pts_per_game, clutch_plus_minus,
    has_estimated_stats, estimate_source_note
FROM player_seasons;
"""

# Columns added after the original schema freeze.
PLAYER_SEASON_EXTRA_COLUMNS = [
    "pts_per_game",
    "fgm_per_game",
    "fga_per_game",
    "fg3m_per_game",
    "fg3a_per_game",
    "ftm_per_game",
    "fta_per_game",
    "oreb_per_game",
    "dreb_per_game",
    "reb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
    "pf_per_game",
    "three_par",
    "ft_rate",
    "ast_to",
]

# Back-compat alias
PLAYER_SEASON_PER_GAME_COLUMNS = PLAYER_SEASON_EXTRA_COLUMNS


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate_db(db_path: Path = DB_PATH) -> None:
    """Apply additive column migrations for databases created before schema additions."""
    view_sql = """
    DROP VIEW IF EXISTS v_player_season_per_game;
    CREATE VIEW v_player_season_per_game AS
    SELECT
        player_id, season, team_id, age, games_played, games_started,
        minutes_total, minutes_per_game,
        pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf,
        CASE WHEN games_played > 0 AND pts  IS NOT NULL THEN ROUND(pts  * 1.0 / games_played, 2) END AS pts_per_game,
        CASE WHEN games_played > 0 AND fgm  IS NOT NULL THEN ROUND(fgm  * 1.0 / games_played, 2) END AS fgm_per_game,
        CASE WHEN games_played > 0 AND fga  IS NOT NULL THEN ROUND(fga  * 1.0 / games_played, 2) END AS fga_per_game,
        CASE WHEN games_played > 0 AND fg3m IS NOT NULL THEN ROUND(fg3m * 1.0 / games_played, 2) END AS fg3m_per_game,
        CASE WHEN games_played > 0 AND fg3a IS NOT NULL THEN ROUND(fg3a * 1.0 / games_played, 2) END AS fg3a_per_game,
        CASE WHEN games_played > 0 AND ftm  IS NOT NULL THEN ROUND(ftm  * 1.0 / games_played, 2) END AS ftm_per_game,
        CASE WHEN games_played > 0 AND fta  IS NOT NULL THEN ROUND(fta  * 1.0 / games_played, 2) END AS fta_per_game,
        CASE WHEN games_played > 0 AND oreb IS NOT NULL THEN ROUND(oreb * 1.0 / games_played, 2) END AS oreb_per_game,
        CASE WHEN games_played > 0 AND dreb IS NOT NULL THEN ROUND(dreb * 1.0 / games_played, 2) END AS dreb_per_game,
        CASE WHEN games_played > 0 AND reb  IS NOT NULL THEN ROUND(reb  * 1.0 / games_played, 2) END AS reb_per_game,
        CASE WHEN games_played > 0 AND ast  IS NOT NULL THEN ROUND(ast  * 1.0 / games_played, 2) END AS ast_per_game,
        CASE WHEN games_played > 0 AND stl  IS NOT NULL THEN ROUND(stl  * 1.0 / games_played, 2) END AS stl_per_game,
        CASE WHEN games_played > 0 AND blk  IS NOT NULL THEN ROUND(blk  * 1.0 / games_played, 2) END AS blk_per_game,
        CASE WHEN games_played > 0 AND tov  IS NOT NULL THEN ROUND(tov  * 1.0 / games_played, 2) END AS tov_per_game,
        CASE WHEN games_played > 0 AND pf   IS NOT NULL THEN ROUND(pf   * 1.0 / games_played, 2) END AS pf_per_game,
        fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct,
        per, usg_pct, ast_pct, orb_pct, drb_pct, trb_pct, stl_pct, blk_pct, tov_pct, pie,
        ows, dws, ws, ws_per_48, obpm, dbpm, bpm, vorp,
        on_court_off_rating, on_court_def_rating, on_court_net_rating,
        off_court_net_rating, net_rating_on_off_diff,
        clutch_pts_per_game, clutch_plus_minus,
        has_estimated_stats, estimate_source_note
    FROM player_seasons;
    """
    conn = get_connection(db_path)
    try:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(player_seasons)").fetchall()
        }
        for col in PLAYER_SEASON_EXTRA_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE player_seasons ADD COLUMN {col} REAL")
        # Ensure new tables from SCHEMA exist (CREATE IF NOT EXISTS is idempotent)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lineup_combos (
                combo_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                season          TEXT NOT NULL,
                team_id         INTEGER NOT NULL,
                combo_size      INTEGER NOT NULL,
                player_id_1     INTEGER NOT NULL,
                player_id_2     INTEGER NOT NULL,
                player_id_3     INTEGER,
                player_id_4     INTEGER,
                player_id_5     INTEGER,
                minutes_played  REAL,
                off_rating      REAL,
                def_rating      REAL,
                net_rating      REAL,
                pace            REAL,
                FOREIGN KEY (team_id) REFERENCES teams(team_id),
                FOREIGN KEY (player_id_1) REFERENCES players(player_id),
                FOREIGN KEY (player_id_2) REFERENCES players(player_id),
                FOREIGN KEY (player_id_3) REFERENCES players(player_id),
                FOREIGN KEY (player_id_4) REFERENCES players(player_id),
                FOREIGN KEY (player_id_5) REFERENCES players(player_id),
                UNIQUE (season, team_id, combo_size, player_id_1, player_id_2, player_id_3, player_id_4, player_id_5)
            );
            CREATE INDEX IF NOT EXISTS idx_lineup_combos_season_team ON lineup_combos(season, team_id, combo_size);
            CREATE TABLE IF NOT EXISTS player_impact_metrics (
                player_id       INTEGER NOT NULL,
                season          TEXT NOT NULL,
                metric_source   TEXT NOT NULL,
                offense         REAL,
                defense         REAL,
                total           REAL,
                wins_added      REAL,
                sample_note     TEXT,
                PRIMARY KEY (player_id, season, metric_source),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            );
            CREATE INDEX IF NOT EXISTS idx_impact_season ON player_impact_metrics(season, metric_source);
            CREATE TABLE IF NOT EXISTS player_tracking_seasons (
                player_id       INTEGER NOT NULL,
                season          TEXT NOT NULL,
                team_id         INTEGER NOT NULL,
                dist_miles      REAL,
                avg_speed       REAL,
                drives          REAL,
                drive_pts       REAL,
                passes_made     REAL,
                secondary_ast   REAL,
                potential_ast   REAL,
                ast_points_created REAL,
                oreb_contest    REAL,
                dreb_contest    REAL,
                PRIMARY KEY (player_id, season, team_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id),
                FOREIGN KEY (team_id) REFERENCES teams(team_id)
            );
            CREATE TABLE IF NOT EXISTS player_roles (
                player_id       INTEGER NOT NULL,
                season          TEXT NOT NULL,
                primary_role    TEXT NOT NULL,
                role_confidence REAL,
                position_versatility REAL,
                PRIMARY KEY (player_id, season),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            );
            """
        )
        conn.executescript(view_sql)
        conn.commit()
    finally:
        conn.close()


def backfill_per_game_stats(db_path: Path = DB_PATH) -> int:
    """Recompute all *_per_game columns from totals / games_played. Returns rows touched."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(
            """
            UPDATE player_seasons SET
                pts_per_game  = CASE WHEN games_played > 0 AND pts  IS NOT NULL THEN ROUND(pts  * 1.0 / games_played, 2) END,
                fgm_per_game  = CASE WHEN games_played > 0 AND fgm  IS NOT NULL THEN ROUND(fgm  * 1.0 / games_played, 2) END,
                fga_per_game  = CASE WHEN games_played > 0 AND fga  IS NOT NULL THEN ROUND(fga  * 1.0 / games_played, 2) END,
                fg3m_per_game = CASE WHEN games_played > 0 AND fg3m IS NOT NULL THEN ROUND(fg3m * 1.0 / games_played, 2) END,
                fg3a_per_game = CASE WHEN games_played > 0 AND fg3a IS NOT NULL THEN ROUND(fg3a * 1.0 / games_played, 2) END,
                ftm_per_game  = CASE WHEN games_played > 0 AND ftm  IS NOT NULL THEN ROUND(ftm  * 1.0 / games_played, 2) END,
                fta_per_game  = CASE WHEN games_played > 0 AND fta  IS NOT NULL THEN ROUND(fta  * 1.0 / games_played, 2) END,
                oreb_per_game = CASE WHEN games_played > 0 AND oreb IS NOT NULL THEN ROUND(oreb * 1.0 / games_played, 2) END,
                dreb_per_game = CASE WHEN games_played > 0 AND dreb IS NOT NULL THEN ROUND(dreb * 1.0 / games_played, 2) END,
                reb_per_game  = CASE WHEN games_played > 0 AND reb  IS NOT NULL THEN ROUND(reb  * 1.0 / games_played, 2) END,
                ast_per_game  = CASE WHEN games_played > 0 AND ast  IS NOT NULL THEN ROUND(ast  * 1.0 / games_played, 2) END,
                stl_per_game  = CASE WHEN games_played > 0 AND stl  IS NOT NULL THEN ROUND(stl  * 1.0 / games_played, 2) END,
                blk_per_game  = CASE WHEN games_played > 0 AND blk  IS NOT NULL THEN ROUND(blk  * 1.0 / games_played, 2) END,
                tov_per_game  = CASE WHEN games_played > 0 AND tov  IS NOT NULL THEN ROUND(tov  * 1.0 / games_played, 2) END,
                pf_per_game   = CASE WHEN games_played > 0 AND pf   IS NOT NULL THEN ROUND(pf   * 1.0 / games_played, 2) END,
                minutes_per_game = CASE
                    WHEN games_played > 0 AND minutes_total IS NOT NULL
                    THEN ROUND(minutes_total * 1.0 / games_played, 2)
                    ELSE minutes_per_game END
            """
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables/indexes if they don't already exist. Safe to re-run."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    migrate_db(db_path)
    seed_stat_availability(db_path)


def seed_stat_availability(db_path: Path = DB_PATH) -> None:
    """
    Pre-populate stat_availability with known NBA/ABA history milestones.

    This does NOT mean the scraper should skip fetching for unavailable
    seasons, it means downstream code (feature engineering, the model) can
    query this table before treating a NULL column as a scrape failure vs.
    "this stat genuinely didn't exist yet." A row here means "available FROM
    this season onward."

    Note on estimates: Basketball-Reference publishes reconstructed
    estimates for some pre-1973-74 players (e.g. approximate blocks/steals
    and BPM for a handful of dominant players like Wilt Chamberlain, built
    from play-by-play reconstructions and contemporary reporting). Where
    such estimates exist, treat the stat as available but flag the specific
    player_season row via has_estimated_stats / estimate_source_note rather
    than marking the whole season unavailable here.
    """
    milestones = [
        # (stat_category, first_available_season, note)
        ("FIELD_GOALS_BASIC",   "1946-47", "Available since the league's founding (BAA/NBA)."),
        ("FREE_THROWS",         "1946-47", "Available since founding."),
        ("REBOUNDS_TOTAL",      "1950-51", "Total rebounds tracked; no offensive/defensive split yet."),
        ("REBOUNDS_OFF_DEF_SPLIT", "1973-74", "Offensive/defensive rebound split begins."),
        ("ASSISTS",             "1946-47", "Available since founding, though early tracking was inconsistent."),
        ("STEALS_BLOCKS",       "1973-74", "Official steals and blocks tracking begins. Some earlier estimates exist for individual star players via Basketball-Reference; store as has_estimated_stats=1 on those rows rather than leaving season fully blank."),
        ("TURNOVERS",           "1977-78", "Official turnover tracking begins."),
        ("THREE_POINT",         "1979-80", "The 3-point line is introduced league-wide this season."),
        ("ADVANCED_RATE_STATS", "1973-74", "PER/BPM/Win Shares/usage-type stats depend on steals+blocks+turnovers as inputs, so full confidence starts here; earlier seasons may have partial or estimated versions only."),
        ("PLAY_BY_PLAY_DERIVED", "1996-97", "Play-by-play logs (used for on-court/off-court and some clutch stats) become reliably available."),
        ("ON_OFF_LINEUPS",      "2007-08", "5-man lineup and on/off court data becomes available via stats.nba.com."),
        ("PLAYER_TRACKING",     "2013-14", "SportVU/second-spectrum tracking data (speed, distance, touches) becomes available; not currently in this schema but noted for future expansion."),
        ("THREE_PAR_FT_RATE",   "1946-47", "FTr (FTA/FGA) derivable since founding; 3PAr only meaningful from 1979-80 (gate with THREE_POINT)."),
        ("AST_TO_RATIO",        "1977-78", "AST/TO requires official turnover tracking."),
        ("ADVANCED_BPM_VORP",   "1973-74", "BPM/VORP published by Basketball-Reference from 1973-74 onward."),
        ("PUBLIC_IMPACT_RAPM",  "1996-97", "Modern RAPM/DARKO/LEBRON-style metrics require play-by-play; sparse video-reconstructed RAPM exists for select 1980s–90s seasons via third parties."),
    ]

    conn = get_connection(db_path)
    try:
        conn.executemany(
            """
            INSERT OR IGNORE INTO stat_availability (season, stat_category, available, note)
            VALUES (?, ?, 1, ?)
            """,
            [(season, category, note) for category, season, note in milestones],
        )
        conn.commit()
    finally:
        conn.close()


def is_stat_available(stat_category: str, season: str, db_path: Path = DB_PATH) -> bool:
    """
    Check whether a stat category is expected to exist for a given season.

    stat_availability stores one threshold row per category ("first season
    this stat exists"). Season strings like "1973-74" sort correctly with
    plain string comparison since the leading year is always 4 digits, so
    no date parsing is needed.

    Returns True if season >= the recorded threshold season for that
    category, False if the category is unknown or the season predates it.
    Downstream code should call this before treating a NULL column as
    missing/broken data versus "this stat simply doesn't exist yet."
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT season FROM stat_availability WHERE stat_category = ?",
            (stat_category,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    threshold_season = row[0]
    return season >= threshold_season


if __name__ == "__main__":
    init_db()
    n = backfill_per_game_stats()
    print(f"Database initialized at {DB_PATH}")
    print(f"Backfilled per-game stats on {n} player_season rows")
