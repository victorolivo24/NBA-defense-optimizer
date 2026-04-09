"""Database engine and session helpers."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .schema import Base

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "processed" / "nba_defense.sqlite"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"


def create_session_factory(database_url: str = DEFAULT_DATABASE_URL) -> sessionmaker:
    """Create a SQLAlchemy session factory for the configured database."""
    engine = create_engine(database_url, future=True)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def initialize_database(database_url: str = DEFAULT_DATABASE_URL) -> None:
    """Create all tables if they do not already exist."""
    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
