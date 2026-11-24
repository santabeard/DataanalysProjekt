import pyarrow.parquet as pq
import pandas as pd

path = "synth_subset_5k/swed_ed_core.parquet"

pf = pq.ParquetFile(path)

#print(pf.schema_arrow)

sample = pf.read_row_group(0).to_pandas()

#print(sample.sumary())

"""
df = pd.read_parquet(path)
print(df.shape)
print(df.dtypes)
print(df.isna().sum())
print(df.describe(include="all"))
"""

# 4. Column-selective read (much cheaper than full load for wide tables)

cols = pf.schema_arrow.names[:5]
df_subset = pd.read_parquet(path, columns=cols)

# 5. Explore a partitioned directory of parquet files
import pyarrow.dataset as ds
dataset = ds.dataset("data_dir/", format="parquet")
print(dataset.schema)
table = dataset.to_table(columns=["col_a", "col_b"])  # lazy, pushdown filtering
df2 = table.to_pandas()
