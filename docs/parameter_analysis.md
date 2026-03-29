# Pipeline Parameter Analysis

This document explains each parameter in `config.json` under the `pipeline` key,
its purpose, and the effect of changing its value on the analysis results.

---

## `null_threshold`

**Default:** `0.3`

### What it does

After loading the raw CSV, `01_ingest.py` computes the fraction of null (missing) values
in each column. Any column with a null fraction **greater than** `null_threshold` is
dropped from the dataset entirely before further processing.

The Diabetes 130-US Hospitals dataset contains several columns with high missingness,
most notably `weight`, `payer_code`, and `medical_specialty`.

### Effect by value

| Value | Columns dropped (approx.) | Impact |
|-------|---------------------------|--------|
| `0.1` | 5–6 columns (weight, payer_code, medical_specialty + others) | More aggressive; removes potentially useful features |
| `0.3` | 3 columns (weight ≈ 97%, payer_code ≈ 40%) | **Default — balanced trade-off** |
| `0.5` | 1–2 columns (only weight at 97%) | Conservative; keeps partially-filled columns |
| `0.9` | 0 columns | No columns removed; downstream imputation required |

### Recommendation

Use `0.3` as a starting point. Lower to `0.1` if you want a fully complete feature set
for machine learning. Increase to `0.5` if domain knowledge suggests a column with
40–50% missingness is informative.

---

## `outlier_zscore`

**Default:** `3.0`

### What it does

After null handling, rows where **any numeric column** has an absolute Z-score
greater than `outlier_zscore` are removed. Z-scores are computed column-wise
(with column medians used to fill remaining NaNs before scoring).

Affected numeric columns in this dataset include:
`time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`,
`number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`.

### Effect by value

| Value | Rows removed (approx.) | Impact |
|-------|------------------------|--------|
| `2.0` | ~5,000 rows (5%) | Aggressive; may remove clinically extreme but valid cases |
| `2.5` | ~2,500 rows (2.5%) | Moderately strict |
| `3.0` | ~500–1,000 rows (0.5–1%) | **Default — conservative** |
| `4.0` | ~50–100 rows (<0.1%) | Very permissive; only removes extreme anomalies |

### Recommendation

Use `3.0` for general analysis. For survival/readmission modelling, consider `2.5`
to produce cleaner distributions. For clinical research, use `4.0` or disable
outlier removal entirely, as extreme values may represent the most critical patients.

### Distribution effect

At `zscore=3.0`, the resulting `time_in_hospital` distribution is nearly symmetric
around ~4.4 days. Lowering to `2.0` truncates the right tail, removing patients
with stays of 10+ days (who are disproportionately readmitted).

---

## `age_bins`

**Default:** `[0, 30, 50, 70, 100]`

### What it does

The raw dataset encodes age as decade brackets such as `[30-40)`, `[60-70)`.
`01_ingest.py` extracts the midpoint of each bracket and then applies `pd.cut`
with these bins to produce a coarser `age_group` label for analysis.

### Effect by configuration

**Default — 4 groups:**
```
bins:   [0, 30, 50, 70, 100]
labels: ["0-30", "30-50", "50-70", "70+"]
```
Produces broad demographic segments. Each group contains tens of thousands of encounters.

**Finer — 5 groups:**
```
bins:   [0, 20, 40, 60, 80, 100]
labels: ["0-20", "20-40", "40-60", "60-80", "80+"]
```
Separates young adults (20–40) from middle-aged (40–60), revealing a more gradual
increase in readmission risk with age.

**Decade-level — 8 groups:**
```
bins:   [0, 10, 20, 30, 40, 50, 60, 70, 100]
labels: ["<10","10s","20s","30s","40s","50s","60s","70+"]
```
Maximally granular; some groups (e.g. `<10`) will have very few encounters
and unstable readmission rate estimates.

### Recommendation

Use the default 4-group configuration for summary visualisations. Use the
5-group or decade-level configuration when investigating age as a continuous
predictor in regression models.

---

## `readmission_binary`

**Default:** `false`

### What it does

The readmission target variable has three classes:
- `<30` — readmitted within 30 days (clinically significant)
- `>30` — readmitted after 30 days
- `NO` — not readmitted

When `readmission_binary = true`, the query layer (`03_query.py`) collapses
`<30` and `>30` into a single class `1` (readmitted), and maps `NO` → `0`.

### Three-class vs binary analysis

| Mode | Classes | Use case |
|------|---------|----------|
| `false` (3-class) | `<30`, `>30`, `NO` | When early readmission (<30 days) is the primary outcome |
| `true` (binary) | `1`, `0` | When any readmission is the outcome; simpler model evaluation |

### Impact on charts

With `readmission_binary=false`, bar charts show three stacked/grouped segments.
With `readmission_binary=true`, charts simplify to two bars (readmitted / not readmitted),
which is more readable for stakeholder presentations but loses the distinction between
early and late readmissions.

### Clinical note

The `<30` class is the clinically important one — CMS (Centers for Medicare and Medicaid
Services) penalties in the US apply specifically to 30-day readmissions. For policy
analysis, keep `readmission_binary=false` and focus on `<30`.

---

## `top_n_diagnoses`

**Default:** `10`

### What it does

Controls how many ICD-9 primary diagnosis codes are displayed in the top-diagnoses
chart (`fig_04_top_diagnoses`) and returned by `top_diagnoses_by_readmission()`.

### Effect by value

| Value | Chart width | Information density | Common codes shown |
|-------|------------|--------------------|--------------------|
| `5`   | Compact    | Low — only dominant codes | 250 (diabetes), 401 (hypertension) |
| `10`  | **Default** | Balanced | + circulatory (428, 414, 427) |
| `20`  | Wide       | High | + respiratory, renal codes |
| `30`  | Very wide  | Very high | Long tail of rare codes |

### Recommendation

Use `10` for summary reports. Increase to `20` for clinical deep-dives into
diagnostic co-occurrence patterns.

---

## `palette`

**Default:** `"viridis"`

### What it does

Sets the Matplotlib/Seaborn colour palette and the Plotly colour scale used
across all six figures. The value is passed directly to:
- `sns.color_palette(palette, n_colors)` for Matplotlib bar charts
- `color_continuous_scale=palette` in Plotly Express figures

### Available options

| Palette | Type | Best for |
|---------|------|---------|
| `"viridis"` | Sequential (blue→yellow) | **Default** — accessible, print-friendly |
| `"plasma"` | Sequential (purple→yellow) | High contrast |
| `"magma"` | Sequential (black→yellow) | Dark backgrounds |
| `"Blues"` | Sequential (blue) | Single-metric charts |
| `"RdYlGn"` | Diverging (red→green) | Good/bad comparisons |
| `"tab10"` | Qualitative | Categorical data with many classes |
| `"Set2"` | Qualitative | Soft pastel categorical |

### Recommendation

Use `"viridis"` for accessibility (colour-blind safe). Switch to `"tab10"` or `"Set2"`
for charts with many categorical groups (e.g., 10 readmission diagnoses) where
sequential palettes can make similar values hard to distinguish.
