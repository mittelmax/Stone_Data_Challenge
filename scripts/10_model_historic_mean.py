from stone_data_challenge.config import data_dir
import pandas as pd

# # Reading dataframe
df = pd.read_csv(f"{data_dir}/data/clean/spine_stone_ipea.csv")
df = df[["id", "mes_referencia", "TPV_mensal"]]

# # Setting mes_referencia as index
df["mes_referencia"] = pd.to_datetime(df["mes_referencia"])
df = df.sort_values(["id", "mes_referencia"])
df = df[df["mes_referencia"] < "2020-08-30"]
df = df.drop(df.groupby("id").tail(5).index, axis=0).set_index("mes_referencia")

# # Generating historic mean
df["TPV_historic_mean"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(50, 2).mean())
df = df.groupby("id").tail(1).drop("TPV_mensal", axis=1)
df = df[pd.notnull(df["TPV_historic_mean"])]

# # Saving as csv
df.to_csv(f"{data_dir}/data/model/performance/historic_mean_test.csv")