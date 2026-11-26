"""Death file: one row per patient with a recorded death.

In:  sos_deathreg_wide (lopnr_swed, date_death_minimp, inexact_death_date)
Out: OUT_DIR/deaths.parquet
     lopnr_swed          int64
     date_of_death       date   = date_death_minimp (v2 decision: never later than the true date,
                                  so no death that could fall within 30 days is missed)
     inexact_death_date  bool   TRUE for the ~0.8% where day/month were imputed

A patient listed more than once keeps the earliest date (the count is logged).

    python 01_deaths.py
"""
import pandas as pd
import ingest_common as C

COLUMNS = ["lopnr_swed", "date_death_minimp", "inexact_death_date"]


def main():
    C.check_settings()
    path = C.source_path(C.DEATHREG)
    C.log(f"reading {path}")

    parts = []
    n_read = n_no_date = n_excluded = 0
    for b in C.iter_batches(path, COLUMNS):
        n_read += len(b)
        has_date = b["date_death_minimp"].notna()
        n_no_date += int((~has_date).sum())
        b = b.loc[has_date].copy()
        b["lopnr_swed"] = C.to_lopnr(b["lopnr_swed"])
        keep = C.not_excluded(b)
        n_excluded += int((~keep).sum())
        parts.append(b.loc[keep])

    d = pd.concat(parts, ignore_index=True)
    n_rows = len(d)
    d = d.sort_values("date_death_minimp", kind="mergesort").drop_duplicates("lopnr_swed", keep="first")

    out = pd.DataFrame({
        "lopnr_swed": d["lopnr_swed"].astype("int64"),
        "date_of_death": d["date_death_minimp"].dt.date,
        "inexact_death_date": d["inexact_death_date"].fillna(False).astype(bool),
    }).sort_values("lopnr_swed")

    C.log(f"rows read {n_read:,} | no death date {n_no_date:,} | excluded {n_excluded:,} | "
          f"duplicate patients dropped {n_rows - len(out):,}")
    C.log(f"deaths {len(out):,} | inexact dates {int(out['inexact_death_date'].sum()):,} | "
          f"range {d['date_death_minimp'].min():%Y-%m-%d} to {d['date_death_minimp'].max():%Y-%m-%d}")
    C.write_parquet(out, "deaths.parquet")


if __name__ == "__main__":
    main()
