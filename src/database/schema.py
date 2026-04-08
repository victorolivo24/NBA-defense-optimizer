"""SQLAlchemy schema for the lineup-dependent defense optimizer."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Player(Base):
    """Core player identity table."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nba_player_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    team_abbreviation: Mapped[str | None] = mapped_column(String(8), nullable=True)
    position: Mapped[str | None] = mapped_column(String(16), nullable=True)
    height: Mapped[str | None] = mapped_column(String(16), nullable=True)
    weight: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    defensive_play_types: Mapped[list["DefensivePlayType"]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )
    lineup_links: Mapped[list["LineupPlayer"]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )


class DefensivePlayType(Base):
    """Player-level defensive play-type metrics such as PPP allowed."""

    __tablename__ = "defensive_play_types"
    __table_args__ = (
        UniqueConstraint(
            "player_id",
            "season",
            "play_type",
            name="uq_defensive_play_type_player_season_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False, index=True)
    season: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    play_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    possessions: Mapped[float | None] = mapped_column(Float, nullable=True)
    points_allowed: Mapped[float | None] = mapped_column(Float, nullable=True)
    ppp_allowed: Mapped[float | None] = mapped_column(Float, nullable=True)
    frequency_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="nba_api")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    player: Mapped["Player"] = relationship(back_populates="defensive_play_types")


class LineupMetric(Base):
    """Lineup-level outcomes and scheme metadata."""

    __tablename__ = "lineup_metrics"
    __table_args__ = (
        UniqueConstraint("lineup_key", "season", name="uq_lineup_key_season"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lineup_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    season: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    minutes_played: Mapped[float | None] = mapped_column(Float, nullable=True)
    possessions: Mapped[float | None] = mapped_column(Float, nullable=True)
    defensive_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    opponent_ppp: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommended_scheme: Mapped[str | None] = mapped_column(String(32), nullable=True)
    observed_scheme: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    players: Mapped[list["LineupPlayer"]] = relationship(
        back_populates="lineup", cascade="all, delete-orphan"
    )


class LineupPlayer(Base):
    """Association table linking a lineup to its five player slots."""

    __tablename__ = "lineup_players"
    __table_args__ = (
        UniqueConstraint("lineup_id", "player_id", name="uq_lineup_player"),
        UniqueConstraint("lineup_id", "slot", name="uq_lineup_slot"),
        CheckConstraint("slot >= 1 AND slot <= 5", name="ck_lineup_slot_range"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lineup_id: Mapped[int] = mapped_column(ForeignKey("lineup_metrics.id"), nullable=False, index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False, index=True)
    slot: Mapped[int] = mapped_column(Integer, nullable=False)

    lineup: Mapped["LineupMetric"] = relationship(back_populates="players")
    player: Mapped["Player"] = relationship(back_populates="lineup_links")
