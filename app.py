"""
Vercel / local entrypoint.

Re-exports the Flask `app` from web/app.py so zero-config Flask detection
finds a top-level instance named `app`. Run locally with: python app.py
"""

from web.app import app, get_engine

__all__ = ["app"]


if __name__ == "__main__":
    get_engine()
    app.run(debug=True, port=5050)
