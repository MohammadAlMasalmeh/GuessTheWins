# GuessTheWins

Guess how many wins a random five of NBA players would take in an 82-game season.

Play daily (same five boards worldwide) or free play at [guessthewins.vercel.app](https://guessthewins.vercel.app).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://127.0.0.1:5050](http://127.0.0.1:5050).

## What's in this repo

- `web/` — Flask app, UI, and slim runtime DB
- `game/` — roster dealing and scoring
- `model/` — win predictor (numpy Stage 1 + Monte Carlo season sims)
- `features/` — roster feature composition used at inference
