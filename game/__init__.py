"""
game/

Game logic for GuessTheWins, built on top of the Stage 1/2 win-prediction
model (model/simulate.py) and the scraped player database (db.py).

- game/roster.py: fame-weighted player-season sampling (difficulty knob).
- game/engine.py: rounds of 5 rosters, guess collection, scoring.

No UI here by design — this is the engine a CLI/web/API layer wires up.
"""
