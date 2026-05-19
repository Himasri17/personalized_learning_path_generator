"""
run.py
------
Flask development server entry point.

Usage
-----
    python run.py                        # default: development, port 5000
    APP_ENV=production python run.py     # production config (still uses Flask's server — not for real prod)

For production, use Gunicorn via Docker / docker-compose:
    gunicorn "app:create_app()" -w 4 -k gevent --bind 0.0.0.0:5000

Environment variables (see .env.example)
-----------------------------------------
APP_ENV     development | testing | production   (default: development)
HOST        bind address                          (default: 0.0.0.0)
PORT        listen port                           (default: 5000)
"""

from __future__ import annotations

import os

from app import create_app

# ---------------------------------------------------------------------------
# Load .env if python-dotenv is installed (dev convenience)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # env-vars already set in shell take priority
except ImportError:
    pass   # dotenv is optional; in production the env is set by Docker / the OS

# ---------------------------------------------------------------------------
# Create app
# ---------------------------------------------------------------------------
env = os.getenv("APP_ENV", "development")
app = create_app(env)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))

    # use_reloader=True auto-restarts on code changes.
    # use_debugger=True enables the Werkzeug interactive debugger.
    # Both are safe only in development — config guards this.
    app.run(
        host        = host,
        port        = port,
        debug       = app.config.get("DEBUG", False),
        use_reloader= app.config.get("DEBUG", False),
    )