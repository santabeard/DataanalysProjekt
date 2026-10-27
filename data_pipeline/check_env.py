"""Check that Python and the packages used by 01-03 work here. Takes a few seconds."""
import os, sys, tempfile
from pathlib import Path

ok = True
def check(name, fn):
    global ok
    try:
        out = fn()
        print(f"  OK    {name}" + (f": {out}" if out else ""))
    except Exception as e:
        ok = False
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")

print("Python", sys.version.split()[0])
check("python >= 3.9", lambda: None if sys.version_info >= (3, 9) else 1 / 0)
for mod in ["argparse", "csv", "re", "time", "collections"]:
    check(f"import {mod}", lambda m=mod: __import__(m) and None)
check("import numpy", lambda: __import__("numpy").__version__)
check("import pandas", lambda: __import__("pandas").__version__)
check("import pyarrow", lambda: __import__("pyarrow").__version__)
for mod in ["pyarrow.compute", "pyarrow.dataset", "pyarrow.parquet"]:
    check(f"import {mod}", lambda m=mod: __import__(m) and None)


def features():
    import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.dataset as ds, pyarrow.parquet as pq
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.parquet"
        t = pa.table({"lopnr_swed": [1.0, 2.0, 3.0], "x": ["a", "b", None],
                      "t": pa.array([0, 60_000_000, 120_000_000], pa.timestamp("us", tz="UTC"))})
        with pq.ParquetWriter(f, t.schema, compression="zstd") as w:          # 01: zstd writer
            w.write_table(t)
        pf = pq.ParquetFile(f, buffer_size=1 << 20, pre_buffer=False)        # 01: buffered streaming read
        b = next(pf.iter_batches(batch_size=2))
        assert b.filter(pc.is_in(b.column("lopnr_swed"), value_set=pa.array([2.0]))).num_rows == 1
        n = sum(x.num_rows for x in ds.dataset(f).to_batches(filter=ds.field("lopnr_swed").isin([1.0, 3.0])))
        assert n == 2                                                       # 02: dataset filter
        df = pq.read_table(f).to_pandas()
        w = pd.DataFrame({"lopnr_swed": [1], "win_start": df["t"].iloc[:1]})
        m = pd.merge_asof(df.astype({"lopnr_swed": "int64"}).sort_values("t"), w, left_on="t", right_on="win_start", by="lopnr_swed")
        assert len(m) == 3                                                  # 02: visit linking
        assert df["x"].astype("string").str.contains("a", regex=True).sum() == 1
        assert pd.Series([1, None], dtype="Int64").isna().sum() == 1
        assert df["t"].dt.tz_convert("Europe/Stockholm").dt.day_of_week.notna().all()
        assert "ehr_x".removeprefix("ehr_") == "x"
        pa.Table.from_pandas(df, preserve_index=False)
    return "zstd, streaming read, dataset filter, merge_asof, string/Int64 dtypes, time zones"


check("features used by the scripts", features)

def need(path, what):
    def f():
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found ({what})")
    return f

def data_files():
    n = len(list(data.glob("*.parquet")))
    if n == 0:
        raise FileNotFoundError(f"no .parquet files in {data}")
    return f"{n} parquet files"

data = Path(os.environ.get("SWED_DATA", "/cygnus/proj/nobackup/swed_db/clean_parquet"))
check(f"data folder ({data})", data_files)
here = Path.cwd()
check("data_info/ in this folder", need(here / "data_info" / "swed_lab_meas_to_snomed.csv", "copy data_info/ here"))
check("out/exclude_lopnr.csv", need(here / "out" / "exclude_lopnr.csv", "run Tolvan's cell in swed_tasks.ipynb; scripts still run without it"))

print("\nALL OK" if ok else "\nSOMETHING FAILED: see FAIL lines above")
