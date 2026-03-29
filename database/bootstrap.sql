-- bootstrap.sql — Minimal schema for the generic data pipeline.
-- Creates only the _datasets registry table.
-- All dataset tables (ds_*) are created dynamically at import time by pipeline.py.

CREATE TABLE IF NOT EXISTS _datasets (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name        TEXT    NOT NULL UNIQUE,
    display_name      TEXT    NOT NULL,
    dataset_type      TEXT    NOT NULL DEFAULT 'generic',
    enrichment_status TEXT    NOT NULL DEFAULT 'none',
    row_count         INTEGER,
    col_count         INTEGER,
    columns           TEXT,
    checksum          TEXT,
    uploaded_at       TEXT    NOT NULL
);
