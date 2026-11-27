#!/bin/bash
#SBATCH -A sens2020598
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -J uu_profile
#SBATCH -o uu_profile_%j.log

# Aggregate-only profile of the uu_data tables, written to the log above.
# Submit from the pipeline folder:   sbatch batch/uu_profile.sbatch
#
# Privacy rules (the log is meant to be exported):
#   - ID-like columns (lopnr, *id, pnr, ...): no values, only counts and overlap rates.
#   - Category values only if >= 10 rows share them; values longer than 40 chars,
#     with 6+ digits in a row, or in free-text/staff columns are never printed.
#   - Formats are shown as shapes (letters -> A/a, digits -> 9), never content.
#   - Numbers/dates: percentiles p1..p99 only, never min/max.
#   - Every count is rounded to the nearest 5.
# Read the log yourself before exporting it.

set -euo pipefail
module load python_ML_packages/3.9.5-cpu
cd "$SLURM_SUBMIT_DIR"

export UU_DIR="${UU_DIR:-$SLURM_SUBMIT_DIR/../../../data/raw/uu_data}"
export SWED_CORE="${SWED_CORE:-$SLURM_SUBMIT_DIR/../../../data/raw/swed_data/copy_swed_ed_core.parquet}"

python3 - <<'PYEOF'
import os, re, sys, itertools
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

MIN = 10          # minimum rows sharing a value before it is printed
ROUND = 5         # every count rounded to this
TOP = 40          # max values printed per column
MAX_VALUE_LEN = 40
ID_RE = re.compile(r"(lopnr|pnr|personn|personid|id$|^id_|idnr|lpnr)", re.I)  # id$ also catches CamelCase PersonId, ContactId
PATIENT_ID_RE = re.compile(r"(lopnr|lpnr|pnr|personn|personid)", re.I)
FREE_TEXT_RE = re.compile(r"(kommentar|comment|fritext|free_?text|anteckning|note|namn$|signatur|personal|staff|"
                          r"nurse|doctor|physician|clinician|lakare|läkare|sjukskot|sjuksköt|ssk|user|anvandare|"
                          r"användare|adress|address|epost|email|telefon|phone|registered_?by|author|sign)", re.I)
LONG_DIGITS_RE = re.compile(r"\d{6,}")  # personnummer-like; bracketed codes like "[29857009]" are allowed
PERSON_NAME_RE = re.compile(r"^[A-ZÅÄÖÉ][a-zåäöé]+(?:[ -][A-ZÅÄÖÉ][a-zåäöé]+)+$")  # "Anna Svensson", "Bo Ek"

uu_dir = Path(os.environ["UU_DIR"])
swed_core = Path(os.environ["SWED_CORE"])

def r(n): return int(round(n / ROUND)) * ROUND
def pct(a, b): return f"{100 * a / b:.1f}%" if b else "n/a"
def out(s=""): print(s, flush=True)

def shape(v):
    s = re.sub(r"[A-ZÅÄÖÉ]", "A", v)
    s = re.sub(r"[a-zåäöé]+", "a", s)
    return re.sub(r"\d", "9", s)

def safe_value(col_name, v):
    v = str(v)
    if (FREE_TEXT_RE.search(col_name) or len(v) > MAX_VALUE_LEN or PERSON_NAME_RE.match(v.strip())
            or LONG_DIGITS_RE.search(re.sub(r"\[\d+\]", "", v))):
        return None
    return v

def value_counts(col):
    """(values, counts) of a ChunkedArray, most common first, nulls excluded."""
    vc = pc.value_counts(col.drop_null()) if col.null_count < len(col) else None
    if vc is None or len(vc) == 0:
        return [], np.array([], dtype=np.int64)
    vals = vc.field("values").to_pylist()
    cnts = np.asarray(vc.field("counts").to_numpy(), dtype=np.int64)
    order = np.argsort(-cnts, kind="stable")
    return [vals[i] for i in order], cnts[order]

def print_top(name, vals, cnts, total):
    shown = 0
    for v, c in zip(vals, cnts):
        if c < MIN or shown >= TOP:
            break
        sv = safe_value(name, v)
        if sv is None:
            continue
        out(f"      {sv!r}: {r(c):,} ({pct(c, total)})")
        shown += 1
    rest = len(vals) - shown
    if rest > 0:
        out(f"      ... {rest:,} other distinct values not shown (rare, long, or unsafe)")

def print_shapes(vals, cnts, total, cap=2_000_000):
    agg = {}
    for v, c in zip(vals[:cap], cnts[:cap]):
        k = shape(str(v))
        agg[k] = agg.get(k, 0) + int(c)
    items = sorted(agg.items(), key=lambda kv: -kv[1])
    shown = [(k, c) for k, c in items if c >= MIN and len(k) <= 60][:15]
    out("    formats (A=upper, a=lower-case word, 9=digit):")
    for k, c in shown:
        out(f"      {k!r}: {r(c):,} ({pct(c, total)})")

def percentiles(arr):
    return np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])

def profile_column(table, name):
    col = table.column(name)
    n = len(col)
    t = col.type
    out(f"  - {name}  [{t}]")
    out(f"    null: {pct(col.null_count, n)}")
    try:
        if ID_RE.search(name):
            vals, cnts = value_counts(col)
            out(f"    ID-like column: values never printed. distinct: {len(vals):,}")
            if len(cnts):
                p = percentiles(cnts)
                out(f"    rows per distinct value p5/p50/p95/p99: {p[1]:.0f} / {p[3]:.0f} / {p[5]:.0f} / {p[6]:.0f}")
            return
        if pa.types.is_boolean(t):
            nn = col.drop_null()
            out(f"    true: {pct(pc.sum(pc.cast(nn, pa.int64())).as_py() or 0, len(nn))}")
        elif pa.types.is_timestamp(t) or pa.types.is_date(t):
            nn = col.drop_null()
            if len(nn) < MIN:
                out("    (too few values)"); return
            unit = t.unit if pa.types.is_timestamp(t) else "D"
            ints = np.asarray(pc.cast(pc.cast(nn, pa.int32()) if pa.types.is_date32(t) else nn, pa.int64()).to_numpy(), dtype=np.int64)  # date32 -> int64 needs int32 first on pyarrow 7
            per_day = {"s": 86400, "ms": 86400_000, "us": 86400_000_000, "ns": 86400_000_000_000, "D": 1}[unit]
            p = percentiles(ints)
            to_month = lambda x: np.datetime64(int(x), unit if unit != "D" else "D").astype("datetime64[M]")
            out(f"    timezone: {getattr(t, 'tz', None)}")
            out(f"    month p1 / p50 / p99: {to_month(p[0])} / {to_month(p[3])} / {to_month(p[6])}")
            if unit != "D":
                midnight = int((ints % per_day == 0).sum())
                out(f"    exactly 00:00:00 UTC (date only?): {pct(midnight, len(ints))}")
                hours = (ints % per_day) // (per_day // 24)
                h = np.bincount(hours.astype(np.int64), minlength=24)
                out("    share per UTC hour 0..23: " + " ".join(f"{100 * x / len(ints):.0f}" for x in h))
        elif pa.types.is_integer(t) or pa.types.is_floating(t):
            nn = col.drop_null()
            vals, cnts = value_counts(col) if len(nn) else ([], [])
            out(f"    distinct: {len(vals):,}")
            if 0 < len(vals) <= 30:
                out("    values (code-like column):")
                print_top(name, vals, cnts, n)
            elif len(nn) >= MIN:
                p = percentiles(np.asarray(nn.to_numpy(), dtype=np.float64))
                out("    p1 / p5 / p25 / p50 / p75 / p95 / p99: " + " / ".join(f"{x:.4g}" for x in p))
        elif pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_dictionary(t):
            if pa.types.is_dictionary(t):
                col = pc.cast(col, pa.string())
            vals, cnts = value_counts(col)
            out(f"    distinct: {len(vals):,}")
            if FREE_TEXT_RE.search(name):
                out("    free-text/staff column: values never printed")
            else:
                out("    most common values:")
                print_top(name, vals, cnts, n)
            print_shapes(vals, cnts, n)
        else:
            out("    (type not profiled)")
    except Exception as e:  # keep going; the error message itself contains no data values
        out(f"    could not profile: {type(e).__name__}")

def id_set(path, name):
    col = pq.read_table(path, columns=[name]).column(name).drop_null()
    try:
        col = pc.cast(col, pa.float64())
    except Exception:
        col = pc.cast(col, pa.string())
    return set(pc.unique(col).to_pylist())

# ---------------------------------------------------------------------------
out(f"python {sys.version.split()[0]} | pyarrow {pa.__version__} | numpy {np.__version__}")
files = sorted(uu_dir.glob("*.parquet"))
out(f"uu_data folder: {len(files)} file(s): {[f.name for f in files]}")
out()

id_cols = {}
for f in files:
    md = pq.read_metadata(f)
    schema = pq.read_schema(f)
    out("=" * 78)
    out(f"FILE {f.name}: {r(md.num_rows):,} rows, {md.num_columns} columns, "
        f"{md.num_row_groups} row group(s), {f.stat().st_size / 1e6:.0f} MB")
    out("SCHEMA:")
    for field in schema:
        out(f"  {field.name}: {field.type}")
    out("COLUMNS:")
    for name in schema.names:
        table = pq.read_table(f, columns=[name])
        profile_column(table, name)
        if ID_RE.search(name):
            id_cols.setdefault(f, []).append(name)
        del table

    # Per-measurement breakdown for the measurements file: which names exist,
    # and for each common name which value columns are filled.
    if "meas" in f.name.lower():
        out("MEASUREMENT BREAKDOWN (per common value of each name-like column):")
        str_cols = [x.name for x in schema if (pa.types.is_string(x.type) or pa.types.is_large_string(x.type)
                    or pa.types.is_dictionary(x.type)) and not ID_RE.search(x.name) and not FREE_TEXT_RE.search(x.name)]
        num_cols = [x.name for x in schema if (pa.types.is_integer(x.type) or pa.types.is_floating(x.type))
                    and not ID_RE.search(x.name)]
        for key in str_cols:
            kcol = pq.read_table(f, columns=[key]).column(key)
            if pa.types.is_dictionary(kcol.type):
                kcol = pc.cast(kcol, pa.string())
            vals, cnts = value_counts(kcol)
            if not (2 <= len(vals) <= 2000):
                continue
            out(f"  by {key}:")
            others = [c for c in str_cols if c != key]
            sub = pq.read_table(f, columns=[key] + num_cols + others)
            if pa.types.is_dictionary(sub.column(key).type):
                sub = sub.set_column(sub.schema.get_field_index(key), key, pc.cast(sub.column(key), pa.string()))
            for v, c in list(zip(vals, cnts))[:TOP]:
                sv = safe_value(key, v)
                if c < MIN or sv is None:
                    continue
                part = sub.filter(pc.equal(sub.column(key), pa.scalar(v, pa.string())))
                out(f"    {sv!r} ({r(c):,} rows)")
                for nc in num_cols:
                    nn = part.column(nc).drop_null()
                    if len(nn) >= MIN:
                        p = percentiles(np.asarray(nn.to_numpy(), dtype=np.float64))
                        out(f"      {nc}: filled {pct(len(nn), len(part))}, p5/p50/p95 {p[1]:.4g} / {p[3]:.4g} / {p[5]:.4g}")
                    else:
                        out(f"      {nc}: filled {pct(len(nn), len(part))}")
                for oc in others:
                    ocol = part.column(oc)
                    if pa.types.is_dictionary(ocol.type):
                        ocol = pc.cast(ocol, pa.string())
                    ovals, ocnts = value_counts(ocol)
                    if not ovals:
                        continue
                    if len(ovals) <= 50:
                        shown = [f"{safe_value(oc, ov)!r}: {r(oc_n):,}" for ov, oc_n in zip(ovals, ocnts)
                                 if oc_n >= MIN and safe_value(oc, ov) is not None][:6]
                        out(f"      {oc}: {', '.join(shown) if shown else '(no common safe values)'}")
                    else:
                        out(f"      {oc}: {len(ovals):,} distinct (filled {pct(len(ocol) - ocol.null_count, len(part))})")
            del sub
    out()

# ---------------------------------------------------------------------------
out("=" * 78)
out("ID LINKING (share of distinct IDs in A that also occur in B; no ID values printed)")
sets = {}
for f, names in id_cols.items():
    for name in names:
        try:
            sets[(f.name, name)] = id_set(f, name)
        except Exception as e:
            out(f"  could not read {f.name}.{name}: {type(e).__name__}")
if swed_core.exists():
    try:
        sets[("SWED " + swed_core.name, "lopnr_swed")] = id_set(swed_core, "lopnr_swed")
    except Exception as e:
        out(f"  could not read SWED lopnr_swed: {type(e).__name__}")
else:
    out(f"  SWED ed_core not found at {swed_core}; skipping SWED overlap")
for (a, b) in itertools.permutations(sets, 2):
    # only meaningful pairs: same column name in two files, or two patient-ID columns
    same_name = a[1] == b[1] and a[0] != b[0]
    both_patient = PATIENT_ID_RE.search(a[1]) and PATIENT_ID_RE.search(b[1]) and a[0] != b[0]
    if not (same_name or both_patient) or not sets[a]:
        continue
    sa, sb = sets[a], sets[b]
    out(f"  {a[0]}.{a[1]} -> {b[0]}.{b[1]}: {pct(len(sa & sb), len(sa))} of {r(len(sa)):,} distinct")
out()
out("done")
PYEOF
