"""
backend/app/blueprints/auth/schemas.py

Marshmallow request/response schemas for the auth blueprint.

All schemas use strict validation so that a single call to
schema.validate() surfaces every field error at once, before any
business logic runs.

Schemas defined here:
  - RegisterSchema            POST /auth/register
  - LoginSchema               POST /auth/login
  - PasswordChangeSchema      POST /auth/password/change
  - PasswordResetRequestSchema  POST /auth/password/reset-request
  - PasswordResetConfirmSchema  POST /auth/password/reset-confirm
  - UserResponseSchema         (serialisation helper, not a request schema)
"""

import re
from marshmallow import Schema, fields, validate, validates, validates_schema, ValidationError, post_load

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Minimum 8 chars, at least one uppercase, one lowercase, one digit, one special char
_PASSWORD_REGEX = re.compile(
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]).{8,}$'
)

_USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_.-]{3,30}$')

PASSWORD_VALIDATORS = [
    validate.Length(min=8, max=128, error="Password must be between 8 and 128 characters."),
]


def _validate_password_strength(value: str) -> None:
    """Shared password-strength validator used across multiple schemas."""
    if not _PASSWORD_REGEX.match(value):
        raise ValidationError(
            "Password must contain at least one uppercase letter, one lowercase letter, "
            "one digit, and one special character (!@#$%^&*...)."
        )


# ---------------------------------------------------------------------------
# RegisterSchema
# ---------------------------------------------------------------------------

class RegisterSchema(Schema):
    """
    Validates the body of POST /auth/register.

    Required fields : username, email, password
    Optional fields : full_name
    """

    username = fields.Str(
        required=True,
        load_default=None,
        metadata={"example": "jane_doe"},
        validate=[
            validate.Length(min=3, max=30, error="Username must be 3–30 characters."),
            validate.Regexp(
                _USERNAME_REGEX,
                error="Username may only contain letters, numbers, underscores, hyphens, and dots.",
            ),
        ],
    )

    email = fields.Email(
        required=True,
        load_default=None,
        metadata={"example": "jane@example.com"},
        validate=validate.Length(max=255, error="Email must be at most 255 characters."),
    )

    password = fields.Str(
        required=True,
        load_default=None,
        load_only=True,        # never serialised in responses
        metadata={"example": "SecurePass123!"},
        validate=PASSWORD_VALIDATORS,
    )

    full_name = fields.Str(
        load_default="",
        validate=validate.Length(max=100, error="Full name must be at most 100 characters."),
        metadata={"example": "Jane Doe"},
    )

    @validates("password")
    def validate_password_strength(self, value):
        _validate_password_strength(value)

    @post_load
    def normalise_email(self, data, **kwargs):
        """Lowercase and strip email before it reaches the DB."""
        data["email"] = data["email"].strip().lower()
        data["username"] = data["username"].strip()
        if data.get("full_name"):
            data["full_name"] = data["full_name"].strip()
        return data


# ---------------------------------------------------------------------------
# LoginSchema
# ---------------------------------------------------------------------------

class LoginSchema(Schema):
    """
    Validates the body of POST /auth/login.

    Accepts email + password only.
    We deliberately do NOT validate password strength here — we rely on
    check_password_hash and want a generic 401 on failure.
    """

    email = fields.Email(
        required=True,
        load_default=None,
        metadata={"example": "jane@example.com"},
    )

    password = fields.Str(
        required=True,
        load_default=None,
        load_only=True,
        validate=validate.Length(min=1, max=128, error="Password cannot be empty."),
    )

    @post_load
    def normalise(self, data, **kwargs):
        data["email"] = data["email"].strip().lower()
        return data


# ---------------------------------------------------------------------------
# PasswordChangeSchema
# ---------------------------------------------------------------------------

class PasswordChangeSchema(Schema):
    """
    Validates the body of POST /auth/password/change.

    Requires the user to supply their current password before setting
    a new one (prevents token-theft account takeover).
    """

    current_password = fields.Str(
        required=True,
        load_only=True,
        validate=validate.Length(min=1, max=128, error="Current password cannot be empty."),
    )

    new_password = fields.Str(
        required=True,
        load_only=True,
        validate=PASSWORD_VALIDATORS,
    )

    @validates("new_password")
    def validate_new_password_strength(self, value):
        _validate_password_strength(value)

    @validates_schema
    def new_differs_from_current(self, data, **kwargs):
        """Prevent users from re-using their current password."""
        if data.get("current_password") and data.get("new_password"):
            if data["current_password"] == data["new_password"]:
                raise ValidationError(
                    "New password must be different from the current password.",
                    field_name="new_password",
                )


# ---------------------------------------------------------------------------
# PasswordResetRequestSchema
# ---------------------------------------------------------------------------

class PasswordResetRequestSchema(Schema):
    """
    Validates the body of POST /auth/password/reset-request.

    Only the email is needed; the backend looks up the user silently.
    """

    email = fields.Email(
        required=True,
        load_default=None,
        validate=validate.Length(max=255),
        metadata={"example": "jane@example.com"},
    )

    @post_load
    def normalise(self, data, **kwargs):
        data["email"] = data["email"].strip().lower()
        return data


# ---------------------------------------------------------------------------
# PasswordResetConfirmSchema
# ---------------------------------------------------------------------------

class PasswordResetConfirmSchema(Schema):
    """
    Validates the body of POST /auth/password/reset-confirm.

    The token is the opaque URL-safe string e-mailed to the user.
    """

    token = fields.Str(
        required=True,
        validate=validate.Length(min=10, max=200, error="Invalid reset token format."),
    )

    new_password = fields.Str(
        required=True,
        load_only=True,
        validate=PASSWORD_VALIDATORS,
    )

    @validates("new_password")
    def validate_new_password_strength(self, value):
        _validate_password_strength(value)


# ---------------------------------------------------------------------------
# UserResponseSchema  (serialisation — not a request schema)
# ---------------------------------------------------------------------------

class UserResponseSchema(Schema):
    """
    Safe outbound representation of a User model instance.

    Usage:
        schema = UserResponseSchema()
        return jsonify(schema.dump(user))
    """

    id          = fields.Str()
    username    = fields.Str()
    email       = fields.Email()
    full_name   = fields.Str()
    created_at  = fields.DateTime(format="iso")
    last_login_at = fields.DateTime(format="iso", allow_none=True)

    class Meta:
        # Explicitly whitelist output fields — password_hash never leaks
        fields = ("id", "username", "email", "full_name", "created_at", "last_login_at")
        dump_only = ("id", "created_at", "last_login_at")