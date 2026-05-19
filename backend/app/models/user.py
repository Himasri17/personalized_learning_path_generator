import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, DateTime, Boolean, Text
)
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash

from ..extensions import db


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = "users"

    # ── Primary key ───────────────────────────────────────────────────────────
    id = Column(String(36), primary_key=True, default=_uuid)

    # ── Identity ──────────────────────────────────────────────────────────────
    email      = Column(String(255), unique=True, nullable=False, index=True)
    full_name  = Column(String(255), nullable=False)
    avatar_url = Column(Text, nullable=True)

    # ── Auth ──────────────────────────────────────────────────────────────────
    password_hash      = Column(String(255), nullable=False)
    is_active          = Column(Boolean, default=True, nullable=False)
    is_email_verified  = Column(Boolean, default=False, nullable=False)
    email_verify_token = Column(String(128), nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at    = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at    = Column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    sessions      = relationship("QuizSession",  back_populates="user",         lazy="dynamic", cascade="all, delete-orphan")
    vark_profiles = relationship("VarkProfile",  back_populates="user",         lazy="dynamic", cascade="all, delete-orphan")
    documents     = relationship("Document",     back_populates="user",         lazy="dynamic", cascade="all, delete-orphan")

    # ── Password helpers ──────────────────────────────────────────────────────
    def set_password(self, raw: str) -> None:
        """Hash and store a plaintext password."""
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        return check_password_hash(self.password_hash, raw)

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self, include_stats: bool = False) -> dict:
        data = {
            "id":                self.id,
            "email":             self.email,
            "full_name":         self.full_name,
            "avatar_url":        self.avatar_url,
            "is_active":         self.is_active,
            "is_email_verified": self.is_email_verified,
            "created_at":        self.created_at.isoformat(),
            "last_login_at":     self.last_login_at.isoformat() if self.last_login_at else None,
        }
        if include_stats:
            sessions   = list(self.sessions)
            completed  = [s for s in sessions if s.status == "completed"]
            avg_score  = (
                round(sum(s.score for s in completed if s.score is not None) / len(completed), 4)
                if completed else None
            )
            data["stats"] = {
                "total_sessions":    len(sessions),
                "completed_sessions": len(completed),
                "avg_score":          avg_score,
            }
        return data

    def __repr__(self) -> str:
        return f"<User {self.email}>"