from stone_data_challenge.config import data_dir
import pandas as pd
from sklearn.metrics import mean_absolute_error
from functools import reduce

# Reading dataframes
df_mean = pd.read_csv(f"{data_dir}/data/model/performance/historic_mean_test.csv").drop("mes_referencia", axis="columns")
df_t1 = pd.read_csv(f"{data_dir}/data/model/performance/test_pred_t1.csv").drop("mes_referencia", axis="columns")
df_t2 = pd.read_csv(f"{data_dir}/data/model/performance/test_pred_t2.csv").drop("mes_referencia", axis="columns")
df_t3 = pd.read_csv(f"{data_dir}/data/model/performance/test_pred_t3.csv").drop("mes_referencia", axis="columns")
df_t4 = pd.read_csv(f"{data_dir}/data/model/performance/test_pred_t4.csv").drop("mes_referencia", axis="columns")
df_t5 = pd.read_csv(f"{data_dir}/data/model/performance/test_pred_t5.csv").drop("mes_referencia", axis="columns")

imp_t1 = pd.read_csv(f"{data_dir}/data/model/performance/importances_t1.csv")
imp_t2 = pd.read_csv(f"{data_dir}/data/model/performance/importances_t2.csv")
imp_t3 = pd.read_csv(f"{data_dir}/data/model/performance/importances_t3.csv")
imp_t4 = pd.read_csv(f"{data_dir}/data/model/performance/importances_t4.csv")
imp_t5 = pd.read_csv(f"{data_dir}/data/model/performance/importances_t5.csv")


# Sortinng values by id
df_t1 = df_t1.sort_values(["id"])
df_t2 = df_t2.sort_values(["id"])
df_t3 = df_t3.sort_values(["id"])
df_t4 = df_t4.sort_values(["id"])
df_t5 = df_t5.sort_values(["id"])

# Merging all performance dataframes
df_comp = pd.merge(df_t1, df_mean, how="inner", on="id")
df_comp = df_comp.sort_values(["id"])
df_comp_2 = pd.merge(df_t2, df_mean, how="inner", on="id")
df_comp_2 = df_comp_2.sort_values(["id"])
df_comp_3 = pd.merge(df_t3, df_mean, how="inner", on="id")
df_comp_3 = df_comp_3.sort_values(["id"])
df_comp_4 = pd.merge(df_t4, df_mean, how="inner", on="id")
df_comp_4 = df_comp_4.sort_values(["id"])
df_comp_5 = pd.merge(df_t5, df_mean, how="inner", on="id")
df_comp_5 = df_comp_5.sort_values(["id"])


data_frames = [df_t1, df_t2, df_t3, df_t4, df_t5, df_mean]

# Using reduce to merge multiple dataframes at once
df_total = reduce(lambda left, right: pd.merge(left, right, on=["id"], how="inner"), data_frames)

df_total["TPV_pred_sum"] = df_total["TPV_pred_t1"] + df_total["TPV_pred_t2"] + df_total["TPV_pred_t3"] +\
        df_total["TPV_pred_t4"] + df_total["TPV_pred_t5"]

df_total["TPV_mean_sum"] = df_total["TPV_historic_mean"] * 5

df_total["TPV_sum"] = df_total["TPV_t1"] + df_total["TPV_t2"] + df_total["TPV_t3"] +\
        df_total["TPV_t4"] + df_total["TPV_t5"]

# Calculating MAE for each prediction
mae_historic_mean_t1 = mean_absolute_error(df_comp["TPV_historic_mean"], df_comp["TPV_t1"])
mae_historic_mean_t2 = mean_absolute_error(df_comp_2["TPV_historic_mean"], df_comp_2["TPV_t2"])
mae_historic_mean_t3 = mean_absolute_error(df_comp_3["TPV_historic_mean"], df_comp_3["TPV_t3"])
mae_historic_mean_t4 = mean_absolute_error(df_comp_4["TPV_historic_mean"], df_comp_4["TPV_t4"])
mae_historic_mean_t5 = mean_absolute_error(df_comp_5["TPV_historic_mean"], df_comp_5["TPV_t5"])
mae_historic_mean_total = mean_absolute_error(df_total["TPV_mean_sum"], df_total["TPV_sum"])

mae_model_t1 = mean_absolute_error(df_comp["TPV_pred_t1"], df_comp["TPV_t1"])
mae_model_t2 = mean_absolute_error(df_comp_2["TPV_pred_t2"], df_comp_2["TPV_t2"])
mae_model_t3 = mean_absolute_error(df_comp_3["TPV_pred_t3"], df_comp_3["TPV_t3"])
mae_model_t4 = mean_absolute_error(df_comp_4["TPV_pred_t4"], df_comp_4["TPV_t4"])
mae_model_t5 = mean_absolute_error(df_comp_5["TPV_pred_t5"], df_comp_5["TPV_t5"])
mae_model_total = mean_absolute_error(df_total["TPV_pred_sum"], df_total["TPV_sum"])

# Saving performance dataframe
df_total.to_csv(f"{data_dir}/data/model/performance/df_performance.csv")
