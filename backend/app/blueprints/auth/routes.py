"""
backend/app/blueprints/auth/routes.py

Blueprint: auth
Endpoints:
  POST /auth/register   — Create a new user account
  POST /auth/login      — Authenticate and issue JWT access + refresh tokens
  POST /auth/refresh    — Rotate access token using a valid refresh token
  POST /auth/logout     — Revoke the current refresh token (server-side blocklist)
  GET  /auth/me         — Return the authenticated user's profile
  POST /auth/password/change  — Change password (authenticated)
  POST /auth/password/reset-request  — Send password-reset email (unauthenticated)
  POST /auth/password/reset-confirm  — Apply new password using reset token
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db, jwt, redis_client
from app.blueprints.auth.schemas import (
    RegisterSchema,
    LoginSchema,
    PasswordChangeSchema,
    PasswordResetRequestSchema,
    PasswordResetConfirmSchema,
)

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# JWT blocklist key prefix in Redis  (value = "1", TTL = token expiry)
_BLOCKLIST_PREFIX = "jwt_blocklist:"


# ---------------------------------------------------------------------------
# JWT token blocklist loader (registered on the jwt extension)
# ---------------------------------------------------------------------------
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload: dict) -> bool:
    jti = jwt_payload["jti"]
    token_in_redis = redis_client.get(f"{_BLOCKLIST_PREFIX}{jti}")
    return token_in_redis is not None


# ---------------------------------------------------------------------------
# POST /auth/register
# ---------------------------------------------------------------------------
@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user.

    Request JSON:
      {
        "username": "jane_doe",
        "email": "jane@example.com",
        "password": "SecurePass123!",
        "full_name": "Jane Doe"         (optional)
      }

    Returns 201 with user info + token pair on success.
    """
    from app.models.user import User

    schema = RegisterSchema()
    errors = schema.validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify({"errors": errors}), 422

    data = schema.load(request.get_json())

    # Uniqueness checks
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "An account with this email already exists."}), 409

    if User.query.filter_by(username=data["username"]).first():
        return jsonify({"error": "Username is already taken."}), 409

    user = User(
        username=data["username"],
        email=data["email"],
        full_name=data.get("full_name", ""),
        password_hash=generate_password_hash(data["password"]),
    )
    db.session.add(user)
    db.session.commit()

    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))

    logger.info("New user registered: %s (%s)", user.username, user.id)

    return jsonify({
        "message": "Account created successfully.",
        "user": _serialize_user(user),
        "access_token": access_token,
        "refresh_token": refresh_token,
    }), 201


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------
@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticate a user and return JWT tokens.

    Request JSON:
      {
        "email": "jane@example.com",
        "password": "SecurePass123!"
      }

    Returns 200 with access_token + refresh_token on success.
    """
    from app.models.user import User

    schema = LoginSchema()
    errors = schema.validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify({"errors": errors}), 422

    data = schema.load(request.get_json())

    user = User.query.filter_by(email=data["email"]).first()

    # Use a constant-time comparison to avoid user-enumeration timing attacks
    if not user or not check_password_hash(user.password_hash, data["password"]):
        return jsonify({"error": "Invalid email or password."}), 401

    if user.is_banned:
        return jsonify({"error": "Your account has been suspended."}), 403

    # Update last-login timestamp
    user.last_login_at = datetime.now(timezone.utc)
    db.session.commit()

    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))

    logger.info("User logged in: %s", user.id)

    return jsonify({
        "message": "Login successful.",
        "user": _serialize_user(user),
        "access_token": access_token,
        "refresh_token": refresh_token,
    }), 200


# ---------------------------------------------------------------------------
# POST /auth/refresh
# ---------------------------------------------------------------------------
@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """
    Issue a new access token using a valid refresh token.

    The old refresh token is NOT rotated here (stateless refresh).
    Rotation (one-time-use refresh tokens) can be enabled by revoking
    the incoming refresh JTI and issuing a new refresh token.

    Header:
      Authorization: Bearer <refresh_token>
    """
    current_user_id = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user_id)

    return jsonify({
        "access_token": new_access_token,
    }), 200


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------
@auth_bp.route("/logout", methods=["POST"])
@jwt_required(verify_type=False)
def logout():
    """
    Revoke the current token (access or refresh) by adding its JTI
    to the Redis blocklist.

    The client should call this twice if it holds both token types,
    or pass the refresh token to also revoke it.

    Header:
      Authorization: Bearer <token>
    """
    jwt_payload = get_jwt()
    jti = jwt_payload["jti"]
    token_type = jwt_payload["type"]  # "access" | "refresh"

    # TTL mirrors the token's remaining validity so Redis auto-cleans
    exp = jwt_payload.get("exp")
    now = int(datetime.now(timezone.utc).timestamp())
    ttl = max(exp - now, 1) if exp else current_app.config.get("JWT_ACCESS_TOKEN_EXPIRES", 900)

    redis_client.setex(f"{_BLOCKLIST_PREFIX}{jti}", ttl, "1")
    logger.info("Token revoked: jti=%s type=%s", jti, token_type)

    return jsonify({"message": "Successfully logged out."}), 200


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_current_user():
    """
    Return the profile of the currently authenticated user.

    Header:
      Authorization: Bearer <access_token>
    """
    from app.models.user import User

    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not user:
        return jsonify({"error": "User not found."}), 404

    return jsonify({"user": _serialize_user(user)}), 200


# ---------------------------------------------------------------------------
# POST /auth/password/change
# ---------------------------------------------------------------------------
@auth_bp.route("/password/change", methods=["POST"])
@jwt_required()
def change_password():
    """
    Change password for an authenticated user.

    Request JSON:
      {
        "current_password": "OldPass123!",
        "new_password":     "NewPass456!"
      }
    """
    from app.models.user import User

    schema = PasswordChangeSchema()
    errors = schema.validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify({"errors": errors}), 422

    data = schema.load(request.get_json())
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)

    if not check_password_hash(user.password_hash, data["current_password"]):
        return jsonify({"error": "Current password is incorrect."}), 401

    user.password_hash = generate_password_hash(data["new_password"])
    db.session.commit()

    logger.info("Password changed for user: %s", user.id)
    return jsonify({"message": "Password updated successfully."}), 200


# ---------------------------------------------------------------------------
# POST /auth/password/reset-request
# ---------------------------------------------------------------------------
@auth_bp.route("/password/reset-request", methods=["POST"])
def password_reset_request():
    """
    Initiate a password reset flow.

    Generates a short-lived reset token (stored in Redis) and sends
    an email via the configured mail service.

    Request JSON:
      { "email": "jane@example.com" }

    Always returns 200 to avoid leaking whether an email is registered.
    """
    import secrets
    from app.models.user import User

    schema = PasswordResetRequestSchema()
    errors = schema.validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify({"errors": errors}), 422

    data = schema.load(request.get_json())
    user = User.query.filter_by(email=data["email"]).first()

    if user:
        token = secrets.token_urlsafe(32)
        ttl = current_app.config.get("PASSWORD_RESET_TOKEN_TTL", 1800)  # 30 min
        redis_client.setex(f"pwd_reset:{token}", ttl, str(user.id))

        # TODO: integrate with your mail service (SendGrid, SES, etc.)
        _send_password_reset_email(user.email, user.username, token)
        logger.info("Password reset requested for user: %s", user.id)

    # Always return 200 to prevent email enumeration
    return jsonify({
        "message": "If that email is registered, a reset link has been sent."
    }), 200


# ---------------------------------------------------------------------------
# POST /auth/password/reset-confirm
# ---------------------------------------------------------------------------
@auth_bp.route("/password/reset-confirm", methods=["POST"])
def password_reset_confirm():
    """
    Apply a new password using a valid reset token.

    Request JSON:
      {
        "token":        "<reset_token_from_email>",
        "new_password": "FreshPass789!"
      }
    """
    from app.models.user import User

    schema = PasswordResetConfirmSchema()
    errors = schema.validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify({"errors": errors}), 422

    data = schema.load(request.get_json())
    token = data["token"]

    user_id = redis_client.get(f"pwd_reset:{token}")
    if not user_id:
        return jsonify({"error": "Reset token is invalid or has expired."}), 400

    user = User.query.get(user_id.decode() if isinstance(user_id, bytes) else user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    user.password_hash = generate_password_hash(data["new_password"])
    db.session.commit()

    # Consume the token immediately (one-time use)
    redis_client.delete(f"pwd_reset:{token}")
    logger.info("Password reset completed for user: %s", user.id)

    return jsonify({"message": "Password has been reset. You may now log in."}), 200


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _serialize_user(user) -> dict:
    """Return a safe public representation of a User ORM object."""
    return {
        "id":           str(user.id),
        "username":     user.username,
        "email":        user.email,
        "full_name":    user.full_name,
        "created_at":   user.created_at.isoformat(),
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


def _send_password_reset_email(email: str, username: str, token: str) -> None:
    """
    Stub — replace with your mail integration (Flask-Mail, SendGrid, SES).

    The reset link should point to your frontend route, e.g.:
      https://app.example.com/reset-password?token=<token>
    """
    reset_url = f"{current_app.config.get('FRONTEND_URL', '')}/reset-password?token={token}"
    logger.debug(
        "STUB: send reset email to %s | url=%s", email, reset_url
    )
    # mail.send_message(
    #     subject="Reset your password",
    #     recipients=[email],
    #     body=f"Hi {username},\n\nReset your password here:\n{reset_url}\n\nExpires in 30 minutes.",
    # )