from stone_data_challenge.config import data_dir
import pandas as pd

# Reading prediction dataframes
pred_august = pd.read_csv(f"{data_dir}/data/predictions/prediction_august.csv", index_col=[0])
pred_september = pd.read_csv(f"{data_dir}/data/predictions/prediction_september.csv", index_col=[0])
pred_october = pd.read_csv(f"{data_dir}/data/predictions/prediction_october.csv", index_col=[0])
pred_november = pd.read_csv(f"{data_dir}/data/predictions/prediction_november.csv", index_col=[0])
pred_december = pd.read_csv(f"{data_dir}/data/predictions/prediction_december.csv", index_col=[0])

# Gettind ids
df_id = pd.read_csv(f"{data_dir}/data/raw/tpv-mensais-treinamento.csv")["id"].drop_duplicates().reset_index(drop=True)

# Concatenating dataframes on x axis
df_prediction = pd.concat([df_id, pred_august, pred_september, pred_october, pred_november, pred_december], axis=1)

# Writing csv
df_prediction.to_csv(f"{data_dir}/data/predictions/final_prediction.csv")
