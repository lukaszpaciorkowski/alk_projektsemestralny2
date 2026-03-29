"""
introspect.py — Live introspection helpers for generating architecture diagrams.

All three public functions return Mermaid source strings that reflect the
actual state of the database and codebase at call time.
"""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APP_ROOT = Path(__file__).parent.parent.parent  # repo root
_APP_DIR = Path(__file__).parent.parent           # app/

_MAX_COLS_FULL = 15      # tables with <= this many cols show all
_MAX_COLS_SHOWN = 10     # when truncating, show this many (plus PK/FK)

# Map SQLite type strings → short Mermaid-friendly names
_TYPE_MAP: dict[str, str] = {
    "integer": "int",
    "bigint": "int",
    "int": "int",
    "real": "float",
    "float": "float",
    "double": "float",
    "text": "text",
    "varchar": "text",
    "blob": "blob",
    "boolean": "bool",
    "bool": "bool",
    "numeric": "float",
    "decimal": "float",
}


def _sqlite_type(raw: str) -> str:
    """Normalise a SQLite column type to a short label."""
    key = raw.lower().split("(")[0].strip()
    return _TYPE_MAP.get(key, raw.lower() or "text")


def _safe_id(name: str) -> str:
    """Make a name safe for use as a Mermaid node id."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


# ---------------------------------------------------------------------------
# ER diagram
# ---------------------------------------------------------------------------

def generate_er_mermaid(engine: Engine) -> str:
    """
    Generate a Mermaid erDiagram from the live SQLite schema.

    - Introspects all tables via sqlite_master + PRAGMA table_info/foreign_key_list
    - For tables with >_MAX_COLS_FULL columns, shows first _MAX_COLS_SHOWN plus
      any PK/FK columns not already included, then appends a comment with the count
    - Emits relationship lines for FK constraints
    - Groups tables: registry (_datasets, _reports) first, then ds_* tables
    """
    with engine.connect() as conn:
        table_rows = conn.execute(
            text(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ).fetchall()
        table_names: list[str] = [r[0] for r in table_rows]

        # Collect columns and FKs per table
        table_cols: dict[str, list[tuple]] = {}   # name → [(name, type, pk)]
        table_fks: dict[str, list[tuple]] = {}    # name → [(from_col, to_table, to_col)]

        for tname in table_names:
            cols = conn.execute(
                text(f"PRAGMA table_info(\"{tname}\")")
            ).fetchall()
            # (cid, name, type, notnull, dflt_value, pk)
            table_cols[tname] = [(c[1], c[2], c[5]) for c in cols]

            fks = conn.execute(
                text(f"PRAGMA foreign_key_list(\"{tname}\")")
            ).fetchall()
            # (id, seq, table, from, to, on_update, on_delete, match)
            table_fks[tname] = [(f[3], f[2], f[4]) for f in fks]

    # Order: registry tables first, then enrichment, then ds_*
    def _sort_key(n: str) -> tuple:
        if n == "_datasets":
            return (0, n)
        if n == "_reports":
            return (1, n)
        if n.endswith("_diag") or n.endswith("_meds"):
            return (3, n)
        if n.startswith("ds_"):
            return (2, n)
        return (4, n)

    ordered = sorted(table_names, key=_sort_key)

    lines: list[str] = ["erDiagram"]

    # ---- Entity blocks ----
    for tname in ordered:
        cols = table_cols[tname]
        fk_from_cols = {fk[0] for fk in table_fks.get(tname, [])}
        pk_cols = {c[0] for c in cols if c[2] > 0}

        # Decide which columns to show
        if len(cols) <= _MAX_COLS_FULL:
            shown = cols
            hidden = 0
        else:
            # Always include PK and FK columns
            priority = {c[0] for c in cols if c[2] > 0 or c[0] in fk_from_cols}
            head = cols[:_MAX_COLS_SHOWN]
            head_names = {c[0] for c in head}
            extra = [c for c in cols if c[0] in priority and c[0] not in head_names]
            shown = head + extra
            hidden = len(cols) - len(shown)

        safe = _safe_id(tname)
        lines.append(f"    {safe} {{")
        for col_name, col_type, is_pk in shown:
            type_str = _sqlite_type(col_type)
            suffix = ""
            if is_pk:
                suffix = " PK"
            elif col_name in fk_from_cols:
                suffix = " FK"
            lines.append(f"        {type_str} {col_name}{suffix}")
        if hidden:
            lines.append(f"        %%  ... and {hidden} more columns")
        lines.append("    }")

    lines.append("")

    # ---- Relationship lines ----
    for tname in ordered:
        for from_col, to_table, to_col in table_fks.get(tname, []):
            if to_table in table_names:
                a = _safe_id(tname)
                b = _safe_id(to_table)
                lines.append(f'    {a} }}o--|| {b} : "{from_col}"')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline flowchart
# ---------------------------------------------------------------------------

def generate_pipeline_mermaid(engine: Engine) -> str:
    """
    Generate a Mermaid flowchart reflecting the actual pipeline state.

    Reads _datasets (counts, types, enrichment) and _reports (item count)
    from the live DB to show real numbers in node labels.
    """
    with engine.connect() as conn:
        ds_rows = conn.execute(
            text(
                "SELECT table_name, display_name, dataset_type, "
                "enrichment_status, row_count FROM _datasets ORDER BY id"
            )
        ).fetchall()

        try:
            n_reports = conn.execute(
                text("SELECT COUNT(*) FROM _reports")
            ).fetchone()[0]
        except Exception:
            n_reports = 0

    n_datasets = len(ds_rows)
    n_diabetes = sum(1 for r in ds_rows if r[2] == "diabetes")
    n_generic = n_datasets - n_diabetes
    n_enriched = sum(1 for r in ds_rows if r[3] == "done")

    # Build dataset subgraph entries (trim display name to fit)
    ds_nodes: list[str] = []
    for i, r in enumerate(ds_rows):
        name = r[1]
        # Shorten long auto-generated display names
        if "/" in name:
            name = Path(name).stem
        name = name.replace("_", " ")
        if len(name) > 28:
            name = name[:26] + ".."
        rows_str = f"{r[4]:,}" if r[4] else "?"
        node_id = f"DS{i}"
        ds_nodes.append(f'        {node_id}["{name}<br/>{rows_str} rows"]')

    ds_block = "\n".join(ds_nodes) if ds_nodes else '        DSEmpty["(no datasets yet)"]'

    enrichment_note = (
        f"{n_enriched} enriched" if n_enriched else "none enriched"
    )

    lines = [
        "flowchart TD",
        '    Upload["📂 CSV Upload / File Import"]',
        "    Upload --> Validate",
        '    Validate["🔍 Validation<br/>null threshold · outlier z-score"]',
        "    Validate --> Detect",
        '    Detect{"🏷️ Type Detection"}',
        f'    Detect -->|"generic ({n_generic})"| Store',
        f'    Detect -->|"diabetes ({n_diabetes})"| Enrich',
        '    Enrich["⚗️ Enrichment<br/>diagnoses · medications"]',
        "    Enrich --> Store",
        f'    Store[("💾 SQLite data.db<br/>{n_datasets} datasets · {enrichment_note}")]',
        "    Store --> Explore",
        "    Store --> Charts",
        "    Store --> Analytics",
        '    Explore["🔍 Data Exploration<br/>filters · preview · download"]',
        '    Charts["📈 Ad Hoc Charts<br/>14 chart types"]',
        '    Analytics["📊 Analytics<br/>24 functions"]',
        "    Analytics --> Reports",
        "    Explore --> Reports",
        "    Charts --> Reports",
        f'    Reports["📄 Reports<br/>{n_reports} item(s)"]',
        "    Reports --> PDF",
        "    Reports --> HTML",
        '    PDF["📥 PDF Export<br/>fpdf2 · Unicode fonts"]',
        '    HTML["🌐 HTML Export<br/>base64 embedded images"]',
        "",
        "    subgraph Datasets",
        ds_block,
        "    end",
        "",
        "    Store -.-> Datasets",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App architecture flowchart
# ---------------------------------------------------------------------------

def generate_app_architecture_mermaid() -> str:
    """
    Generate a Mermaid flowchart from the live codebase structure.

    Introspects app/views/, app/components/, app/core/ to count and name
    the real files; reads REGISTRY size and chart-type count at import time.
    """
    # --- Views ---
    views_dir = _APP_DIR / "views"
    view_files = sorted(views_dir.glob("*.py")) if views_dir.exists() else []
    _PAGE_ICONS = {
        "data_sources": "📂", "exploration": "🔍", "adhoc": "📈",
        "analytics": "📊", "reports": "📄", "architecture": "📐",
    }

    def _page_label(path: Path) -> str:
        stem = path.stem.lstrip("0123456789_")
        for key, icon in _PAGE_ICONS.items():
            if key in stem.lower():
                return f'{icon} {stem.replace("_", " ").title()}'
        return stem.replace("_", " ").title()

    view_nodes = []
    for i, f in enumerate(view_files):
        label = _page_label(f)
        view_nodes.append(f'        V{i}["{label}"]')

    # --- Components ---
    comp_dir = _APP_DIR / "components"
    comp_files = sorted(p for p in comp_dir.glob("*.py") if p.stem != "__init__") \
        if comp_dir.exists() else []

    # Try to read chart type count from the adhoc_charts view
    n_chart_types = 14  # default
    adhoc_path = views_dir / "3_adhoc_charts.py"
    if adhoc_path.exists():
        src = adhoc_path.read_text(encoding="utf-8")
        m = re.search(r"CHART_TYPES\s*=\s*\[([^\]]+)\]", src, re.DOTALL)
        if m:
            n_chart_types = len(re.findall(r'"[^"]+"', m.group(1)))

    def _comp_label(path: Path) -> str:
        stem = path.stem
        extra = ""
        if "chart_builder" in stem:
            extra = f"<br/>{n_chart_types} chart types"
        return f"{stem.replace('_', ' ')}{extra}"

    comp_nodes = [
        f'        C{i}["{_comp_label(f)}"]'
        for i, f in enumerate(comp_files)
    ]

    # --- Core modules ---
    core_dir = _APP_DIR / "core"
    core_files = sorted(
        p for p in core_dir.glob("*.py") if p.stem not in ("__init__",)
    ) if core_dir.exists() else []

    # Try to get REGISTRY size
    n_registry = 0
    try:
        from app.core.registry import REGISTRY  # noqa: PLC0415
        n_registry = len(REGISTRY)
    except Exception:
        pass

    def _core_label(path: Path) -> str:
        stem = path.stem
        extra = ""
        if "registry" in stem and n_registry:
            extra = f"<br/>{n_registry} functions"
        return f"{stem.replace('_', ' ')}{extra}"

    core_nodes = [
        f'        K{i}["{_core_label(f)}"]'
        for i, f in enumerate(core_files)
    ]

    # DB stats
    try:
        from app.core.pipeline import DB_PATH, get_engine  # noqa: PLC0415
        from sqlalchemy import text as _text  # noqa: PLC0415
        _eng = get_engine(DB_PATH)
        with _eng.connect() as _conn:
            _n_ds = _conn.execute(_text("SELECT COUNT(*) FROM _datasets")).fetchone()[0]
        db_label = f"💾 SQLite data.db<br/>{_n_ds} datasets"
    except Exception:
        db_label = "💾 SQLite data.db"

    n_views = len(view_files)
    n_comps = len(comp_files)
    n_core = len(core_files)

    lines = [
        "flowchart TD",
        f'    subgraph UI["Streamlit UI ({n_views} pages)"]',
        *view_nodes,
        "    end",
        "",
        f'    subgraph Components["app/components/ ({n_comps} modules)"]',
        *comp_nodes,
        "    end",
        "",
        f'    subgraph Core["app/core/ ({n_core} modules)"]',
        *core_nodes,
        "    end",
        "",
        f'    DB[("{db_label}")]',
        "",
        "    UI --> Components",
        "    UI --> Core",
        "    Core --> DB",
        "    Components --> Core",
    ]

    return "\n".join(lines)
