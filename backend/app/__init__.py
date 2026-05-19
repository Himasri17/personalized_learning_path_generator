"""
app/__init__.py
---------------
Flask application factory.

Usage
-----
    from app import create_app
    app = create_app()          # uses APP_ENV env-var (default: development)
    app = create_app("testing") # explicit env
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from flask import Flask
from flask_cors import CORS

from .config import config_map, BaseConfig
from .extensions import db, jwt, migrate, celery, redis_client, ma

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register_blueprints(app: Flask) -> None:
    """Import and register every blueprint with its URL prefix."""

    from app.blueprints.auth.routes    import auth_bp
    from app.blueprints.docs.routes    import docs_bp
    from app.blueprints.quiz.routes    import quiz_bp
    from app.blueprints.analysis.routes import analysis_bp
    from app.blueprints.explain.routes import explain_bp

    app.register_blueprint(auth_bp,     url_prefix="/auth")
    app.register_blueprint(docs_bp,     url_prefix="/docs")
    app.register_blueprint(quiz_bp,     url_prefix="/sessions")
    app.register_blueprint(analysis_bp, url_prefix="/sessions")   # GET /sessions/<id>/results
    app.register_blueprint(explain_bp,  url_prefix="/explain")


def _register_error_handlers(app: Flask) -> None:
    """Global JSON error responses so every error returns consistent JSON."""
    from flask import jsonify

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify(error="Bad request", detail=str(e)), 400

    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify(error="Unauthorised"), 401

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify(error="Forbidden"), 403

    @app.errorhandler(404)
    def not_found(e):
        return jsonify(error="Not found"), 404

    @app.errorhandler(422)
    def unprocessable(e):
        return jsonify(error="Unprocessable entity", detail=str(e)), 422

    @app.errorhandler(500)
    def server_error(e):
        logger.exception("Unhandled server error: %s", e)
        return jsonify(error="Internal server error"), 500


def _configure_logging(app: Flask) -> None:
    level = logging.DEBUG if app.debug else logging.INFO
    logging.basicConfig(
        level  = level,
        format = "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt= "%Y-%m-%d %H:%M:%S",
    )
    # Quieten noisy third-party loggers in production
    if not app.debug:
        for noisy in ("urllib3", "botocore", "s3transfer", "faiss"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _init_celery(app: Flask) -> None:
    """
    Push the Flask app context into every Celery worker task so that
    SQLAlchemy / extensions work inside tasks.
    """
    TaskBase = celery.Task

    class ContextTask(TaskBase):  # type: ignore[valid-type]
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return super().__call__(*args, **kwargs)

    celery.Task = ContextTask  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_app(env: Optional[str] = None) -> Flask:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    env : str, optional
        One of "development", "testing", "production".
        Falls back to the APP_ENV environment variable, then "development".
    """
    env = env or os.getenv("APP_ENV", "development")
    cfg = config_map.get(env)
    if cfg is None:
        raise ValueError(
            f"Unknown APP_ENV '{env}'. Choose from: {list(config_map.keys())}"
        )

    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(cfg)

    # ---- Logging -----------------------------------------------------------
    _configure_logging(app)
    logger.info("Starting Personalized Learning Path API [env=%s]", env)

    # ---- CORS --------------------------------------------------------------
    CORS(
        app,
        resources={r"/*": {"origins": app.config.get("CORS_ORIGINS", ["http://localhost:5173"])}},
        supports_credentials=True,
    )

    # ---- Extensions --------------------------------------------------------
    db.init_app(app)
    jwt.init_app(app)
    migrate.init_app(app, db)
    ma.init_app(app)

    # Redis (non-fatal: warn but continue if Redis is unreachable in testing)
    try:
        redis_client.init_app(app)
    except Exception as exc:            # pragma: no cover
        logger.warning("Redis init warning: %s", exc)

    # Celery – configure broker/backend from app config
    celery.conf.update(
        broker_url            = app.config["REDIS_URL"],
        result_backend        = app.config["REDIS_URL"],
        task_serializer       = "json",
        result_serializer     = "json",
        accept_content        = ["json"],
        timezone              = "UTC",
        enable_utc            = True,
        task_track_started    = True,
        task_acks_late        = True,
        worker_prefetch_multiplier = 1,
    )
    _init_celery(app)

    # ---- Blueprints --------------------------------------------------------
    _register_blueprints(app)

    # ---- Error handlers ----------------------------------------------------
    _register_error_handlers(app)

    # ---- Shell context (flask shell) ---------------------------------------
    @app.shell_context_processor
    def _shell_ctx():
        from app.models.user         import User
        from app.models.quiz_session import QuizSession
        from app.models.question     import Question
        from app.models.response     import Response
        from app.models.vark_profile import VarkProfile
        return {
            "db": db, "User": User, "QuizSession": QuizSession,
            "Question": Question, "Response": Response, "VarkProfile": VarkProfile,
        }

    # ---- Health-check route ------------------------------------------------
    @app.get("/health")
    def health():
        from flask import jsonify
        return jsonify(status="ok", env=env), 200

    logger.info("Flask app created successfully.")
    return app