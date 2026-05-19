"""
app/extensions.py
-----------------
Singleton extension objects created here and initialised in the app factory
(app/__init__.py) via  ext.init_app(app).

Import from here throughout the project to avoid circular imports:
    from app.extensions import db, jwt, celery, redis_client
"""

from __future__ import annotations

from celery import Celery
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_redis import FlaskRedis

# ---------------------------------------------------------------------------
# SQLAlchemy ORM
# ---------------------------------------------------------------------------
db = SQLAlchemy()

# ---------------------------------------------------------------------------
# Alembic migrations (init_app receives db as second arg — done in factory)
# ---------------------------------------------------------------------------
migrate = Migrate()

# ---------------------------------------------------------------------------
# JWT authentication
# ---------------------------------------------------------------------------
jwt = JWTManager()

# ---------------------------------------------------------------------------
# Marshmallow serialisation / validation
# ---------------------------------------------------------------------------
ma = Marshmallow()

# ---------------------------------------------------------------------------
# Redis client (used for caching, session data, Celery result backend)
# ---------------------------------------------------------------------------
redis_client = FlaskRedis()

# ---------------------------------------------------------------------------
# Celery (broker URL is injected in the factory after config is loaded)
# ---------------------------------------------------------------------------
celery = Celery(__name__)


# ---------------------------------------------------------------------------
# JWT callbacks
# ---------------------------------------------------------------------------

@jwt.expired_token_loader
def _expired_token_callback(jwt_header, jwt_payload):
    from flask import jsonify
    return jsonify(error="Token has expired", code="token_expired"), 401


@jwt.invalid_token_loader
def _invalid_token_callback(reason):
    from flask import jsonify
    return jsonify(error="Invalid token", detail=reason, code="invalid_token"), 401


@jwt.unauthorized_loader
def _missing_token_callback(reason):
    from flask import jsonify
    return jsonify(error="Authorisation required", detail=reason, code="missing_token"), 401


@jwt.revoked_token_loader
def _revoked_token_callback(jwt_header, jwt_payload):
    from flask import jsonify
    return jsonify(error="Token has been revoked", code="token_revoked"), 401


@jwt.token_in_blocklist_loader
def _check_if_token_revoked(jwt_header, jwt_payload) -> bool:
    """
    Check whether the JTI (JWT ID) has been added to the Redis blocklist.
    Tokens are blocklisted on logout via  auth/routes.py::logout().
    """
    jti = jwt_payload.get("jti")
    if not jti:
        return False
    try:
        return redis_client.get(f"blocklist:{jti}") is not None
    except Exception:
        # If Redis is unreachable, fail open in dev; consider fail-closed in prod.
        return False