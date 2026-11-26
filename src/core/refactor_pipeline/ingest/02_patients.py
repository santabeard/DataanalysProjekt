"""Age and sex file: one row per patient in swed_ed_core.

In:  swed_ed_core (lopnr_swed, ed_arrival_datetime, ed_age, is_male)
Out: OUT_DIR/patients.parquet
     lopnr_swed  int64
     age         float64  ed_age at the patient's FIRST ED visit
     is_male     int8     1 = male, 0 = female, from the first visit

Age changes between visits, so visits.parquet also carries `age` at each visit;
use that one as the model feature. Patients whose is_male differs between visits
are counted in the log (the first visit's value is kept).

    python 02_patients.py
"""
import pandas as pd
import ingest_common as C

COLUMNS = ["lopnr_swed", "ed_arrival_datetime", "ed_age", "is_male"]


def main():
    C.check_settings()
    path = C.source_path(C.ED_CORE)
    C.log(f"reading {path}")

    firsts, sexes = [], []
    n_read = n_excluded = 0
    for b in C.iter_batches(path, COLUMNS):
        n_read += len(b)
        b["lopnr_swed"] = C.to_lopnr(b["lopnr_swed"])
        keep = C.not_excluded(b)
        n_excluded += int((~keep).sum())
        b = b.loc[keep]
        # reduce each batch right away: earliest visit per patient + distinct sex values
        firsts.append(b.sort_values("ed_arrival_datetime", kind="mergesort").drop_duplicates("lopnr_swed"))
        sexes.append(b[["lopnr_swed", "is_male"]].drop_duplicates())

    first = C.keep_first(firsts, "lopnr_swed", "ed_arrival_datetime")
    sex = pd.concat(sexes, ignore_index=True).drop_duplicates()
    n_sex_conflict = int(sex["lopnr_swed"].duplicated().sum())

    out = pd.DataFrame({
        "lopnr_swed": first["lopnr_swed"].astype("int64"),
        "age": first["ed_age"].astype("float64"),
        "is_male": first["is_male"].astype("int8"),
    }).sort_values("lopnr_swed")

    C.log(f"visits read {n_read:,} | excluded {n_excluded:,} | patients {len(out):,}")
    C.log(f"patients with is_male differing between visits: {n_sex_conflict:,}")
    C.log(f"age at first visit: min {out['age'].min():.0f}, median {out['age'].median():.0f}, "
          f"max {out['age'].max():.0f} | male share {out['is_male'].mean():.3f}")
    C.write_parquet(out, "patients.parquet")


if __name__ == "__main__":
    main()
