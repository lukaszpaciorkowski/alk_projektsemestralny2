"""
create_db.py — Create SQLite database from schema.sql using SQLAlchemy.

Usage:
    python database/create_db.py [--config config.json] [--reset]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_engine(config: dict):
    """Create and return a SQLAlchemy engine from config."""
    db_path = config["database"]["path"]
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    logger.info("Connecting to database: %s", url)
    return create_engine(url, echo=False)


def drop_all_tables(engine) -> None:
    """Drop all application tables in dependency order."""
    tables = [
        "medications",
        "diagnosis_encounters",
        "admissions",
        "patients",
        "diagnoses_lookup",
        "admission_types",
        "discharge_types",
    ]
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        for table in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            logger.info("Dropped table: %s", table)
        conn.execute(text("PRAGMA foreign_keys = ON"))
    logger.info("All tables dropped.")


def create_all_tables(engine) -> None:
    """Execute schema.sql to create all tables and indexes."""
    sql_text = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info("Schema applied — %d statements executed.", len(statements))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the diabetes SQLite database from schema.sql."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all existing tables before recreating them.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config)

    if args.reset:
        logger.warning("--reset flag detected. Dropping all tables...")
        drop_all_tables(engine)

    create_all_tables(engine)
    logger.info("Database ready at: %s", config["database"]["path"])


if __name__ == "__main__":
    main()
