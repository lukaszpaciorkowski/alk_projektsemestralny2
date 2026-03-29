"""
create_db.py — Create SQLite database from bootstrap.sql using SQLAlchemy.

Usage:
    python database/create_db.py [--config config.json] [--reset]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BOOTSTRAP_PATH = Path(__file__).parent / "bootstrap.sql"
DEFAULT_DB_PATH = "data/data.db"


def get_engine(db_path: str):
    """Create and return a SQLAlchemy engine."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    logger.info("Connecting to database: %s", url)
    return create_engine(url, echo=False)


def drop_registry(engine) -> None:
    """Drop the _datasets registry table (and all ds_* tables)."""
    with engine.begin() as conn:
        # Drop all ds_* dynamic tables
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ds_%'")
        ).fetchall()
        for (tname,) in rows:
            conn.execute(text(f"DROP TABLE IF EXISTS [{tname}]"))
            logger.info("Dropped table: %s", tname)
        conn.execute(text("DROP TABLE IF EXISTS _datasets"))
        logger.info("Dropped _datasets registry table.")


def create_registry(engine) -> None:
    """Execute bootstrap.sql to create the _datasets registry table."""
    sql_text = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("Bootstrap schema applied — %d statements executed.", len(statements))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the data.db SQLite database from bootstrap.sql."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to data.db (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all existing tables before recreating them.",
    )
    args = parser.parse_args()

    engine = get_engine(args.db)

    if args.reset:
        logger.warning("--reset flag detected. Dropping all tables...")
        drop_registry(engine)

    create_registry(engine)
    logger.info("Database ready at: %s", args.db)


if __name__ == "__main__":
    main()
