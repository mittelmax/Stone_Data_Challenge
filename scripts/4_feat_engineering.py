from stone_data_challenge.config import data_dir
from sklearn.impute import SimpleImputer
import datetime as dt
import numpy as np
import datetime
import pandas as pd

# # Reading dataframa
df = pd.read_csv(f"{data_dir}/data/clean/spine_stone_ipea.csv")
df = df.replace("nan", np.NaN)
df = df.drop_duplicates()


# # # Checking column types
# df.dtypes

# # Fixing types
df[["MCC", "MacroClassificacao", "StoneCreatedDate", "StoneFirstTransactionDate", "segmento",
    "sub_segmento", "persona", "tipo_documento", "Estado", "mes_referencia", "id"]] = df[["MCC",
    "MacroClassificacao", "StoneCreatedDate", "StoneFirstTransactionDate", "segmento", "sub_segmento", "persona",
    "tipo_documento", "Estado", "mes_referencia", "id"]].astype(str)

# # Fixing date formatting
df["StoneCreatedDate"] = pd.to_datetime(df["StoneCreatedDate"]).dt.tz_localize(None)
df["StoneFirstTransactionDate"] = pd.to_datetime(df["StoneFirstTransactionDate"])
df["mes_referencia"] = pd.to_datetime(df["mes_referencia"])

# # Transforming date variables in cardinal numbers
initial_date = dt.datetime(2020, 7, 31)
df["DaysSinceCreation"] = -(df["StoneCreatedDate"] - initial_date).dt.days
df["DaysSinceFirstTrans"] = -(df["StoneFirstTransactionDate"] - initial_date).dt.days

# # Creating separate year and month variables for date imputing
df["YearCreated"] = df["StoneCreatedDate"].dt.year
df["MonthCreated"] = df["StoneCreatedDate"].dt.month
df["YearFirstTrans"] = df["StoneFirstTransactionDate"].dt.year
df["MonthFirstTrans"] = df["StoneFirstTransactionDate"].dt.month
df["AnoReferencia"] = df["mes_referencia"].dt.year
df["MesReferencia"] = df["mes_referencia"].dt.month

# # # Removing unwanted variables
df.drop(["mes", "StoneFirstTransactionDate", "StoneCreatedDate"], axis="columns", inplace=True)

# # Checking for missing data
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({"col_name": df.columns, "pct_missing": percent_missing})

# # Imputing missing categorical data for business type
df["MacroClassificacao"] = df["MacroClassificacao"].fillna("Desconhecido")
df["segmento"] = df["segmento"].fillna("Desconhecido")
df["sub_segmento"] = df["sub_segmento"].fillna("Desconhecido")

# # Imputing missing categorical data for state
df["Estado"] = df["Estado"].fillna("SP")  # # -> São Paulo is by far the most popular ocurrence

# # Missing data imputation
missing_vars = ["ind_varejo", "demissoes", "DaysSinceFirstTrans", "YearFirstTrans", "MonthFirstTrans"]

# Creatinng column transformer
imputer = SimpleImputer(strategy="median")
df[missing_vars] = imputer.fit_transform(df[missing_vars])  # Imputing data

# # Checking for missing data
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({"col_name": df.columns, "pct_missing": percent_missing})

# # #  Feature Engineering
df = df.set_index(["mes_referencia"])

# # We need to add 5 months to the dataset before shifting the columns
df_agosto = df.groupby("id").tail(1)
df_agosto.index = df_agosto.index + datetime.timedelta(days=30)
df_agosto.loc[:, "TPV_mensal"] = np.nan
df_agosto.loc[:, "caixas_papelao"] = np.nan
df_agosto.loc[:, "caixas_papelao_varpct"] = np.nan
df_agosto.loc[:, "ipca_pct"] = np.nan
df_agosto.loc[:, "cambio_dol"] = np.nan
df_agosto.loc[:, "demissoes"] = np.nan
df_agosto.loc[:, "ind_varejo"] = np.nan
df_agosto[["DaysSinceCreation", "DaysSinceFirstTrans"]] = df_agosto[["DaysSinceCreation", "DaysSinceFirstTrans"]] + 30
df_agosto["MesReferencia"] = df_agosto["MesReferencia"] + 1

df_setembro = df_agosto[:]
df_setembro.index = df_setembro.index + datetime.timedelta(days=30)
df_setembro[["DaysSinceCreation", "DaysSinceFirstTrans"]] = df_setembro[["DaysSinceCreation", "DaysSinceFirstTrans"]] + 30
df_setembro["MesReferencia"] = df_setembro["MesReferencia"] + 1

df_outubro = df_setembro[:]
df_outubro.index = df_outubro.index + datetime.timedelta(days=30)
df_outubro[["DaysSinceCreation", "DaysSinceFirstTrans"]] = df_outubro[["DaysSinceCreation", "DaysSinceFirstTrans"]] + 30
df_outubro["MesReferencia"] = df_outubro["MesReferencia"] + 1

df_novembro = df_outubro[:]
df_novembro.index = df_novembro.index + datetime.timedelta(days=30)
df_novembro[["DaysSinceCreation", "DaysSinceFirstTrans"]] = df_novembro[["DaysSinceCreation", "DaysSinceFirstTrans"]] + 30
df_novembro["MesReferencia"] = df_novembro["MesReferencia"] + 1

df_dezembro = df_novembro[:]
df_dezembro.index = df_dezembro.index + datetime.timedelta(days=30)
df_dezembro[["DaysSinceCreation", "DaysSinceFirstTrans"]] = df_dezembro[["DaysSinceCreation", "DaysSinceFirstTrans"]] + 30
df_dezembro["MesReferencia"] = df_dezembro["MesReferencia"] + 1

df = pd.concat([df, df_agosto,df_setembro,df_outubro,df_novembro,df_dezembro], axis=0)
df = df.sort_values(["id", "mes_referencia"])

# # Creating lagged variables
# TPV
df["TPV_lag_1"] = df.groupby("id")["TPV_mensal"].shift(1)
df["TPV_lag_2"] = df.groupby("id")["TPV_mensal"].shift(2)
df["TPV_lag_3"] = df.groupby("id")["TPV_mensal"].shift(3)
df["TPV_lag_4"] = df.groupby("id")["TPV_mensal"].shift(4)
df["TPV_lag_5"] = df.groupby("id")["TPV_mensal"].shift(5)
df["TPV_lag_6"] = df.groupby("id")["TPV_mensal"].shift(6)
df["TPV_lag_12"] = df.groupby("id")["TPV_mensal"].shift(12)

# IPCA
df["IPCA_lag_1"] = df.groupby("id")["ipca_pct"].shift(1)
df["IPCA_lag_2"] = df.groupby("id")["ipca_pct"].shift(2)
df["IPCA_lag_6"] = df.groupby("id")["ipca_pct"].shift(6)
df["IPCA_lag_12"] = df.groupby("id")["ipca_pct"].shift(12)

# caixas_papelao
df["caixas_papelao_lag_1"] = df.groupby("id")["caixas_papelao"].shift(1)
df["caixas_papelao_lag_2"] = df.groupby("id")["caixas_papelao"].shift(2)
df["caixas_papelao_lag_6"] = df.groupby("id")["caixas_papelao"].shift(6)
df["caixas_papelao_lag_12"] = df.groupby("id")["caixas_papelao"].shift(12)

# caixas_papelao_varpct
df["caixas_papelao_varpct_lag_1"] = df.groupby("id")["caixas_papelao_varpct"].shift(1)
df["caixas_papelao_varpct_lag_2"] = df.groupby("id")["caixas_papelao_varpct"].shift(2)
df["caixas_papelao_varpct_lag_6"] = df.groupby("id")["caixas_papelao_varpct"].shift(6)
df["caixas_papelao_varpct_lag_12"] = df.groupby("id")["caixas_papelao_varpct"].shift(12)

# Demissões
df["demissoes_lag_1"] = df.groupby("id")["demissoes"].shift(1)
df["demissoes_lag_2"] = df.groupby("id")["demissoes"].shift(2)
df["demissoes_lag_6"] = df.groupby("id")["demissoes"].shift(6)
df["demissoes_lag_12"] = df.groupby("id")["demissoes"].shift(12)

# ind_varejo
df["ind_varejo_lag_1"] = df.groupby("id")["ind_varejo"].shift(1)
df["ind_varejo_lag_2"] = df.groupby("id")["ind_varejo"].shift(2)
df["ind_varejo_lag_6"] = df.groupby("id")["ind_varejo"].shift(6)
df["ind_varejo_lag_12"] = df.groupby("id")["ind_varejo"].shift(12)

# cambio_dol
df["cambio_dol_lag_1"] = df.groupby("id")["cambio_dol"].shift(1)
df["cambio_dol_lag_2"] = df.groupby("id")["cambio_dol"].shift(2)
df["cambio_dol_lag_6"] = df.groupby("id")["cambio_dol"].shift(6)
df["cambio_dol_lag_12"] = df.groupby("id")["cambio_dol"].shift(12)


# # Creating moving averages
# Creating moving averages for TPV
df["TPV_MA_2"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(2, 2).mean())
df["TPV_MA_3"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(3, 2).mean())
df["TPV_MA_4"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(4, 2).mean())
df["TPV_MA_5"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(5, 2).mean())
df["TPV_MA_6"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(6, 2).mean())
df["TPV_MA_12"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(12, 2).mean())
df["TPV_MA_18"] = df.groupby("id")["TPV_mensal"].transform(lambda x: x.rolling(18, 2).mean())

# Creating moving averages for IPCA
df["ipca_MA_2"] = df.groupby("id")["ipca_pct"].transform(lambda x: x.rolling(2, 2).mean())
df["ipca_MA_3"] = df.groupby("id")["ipca_pct"].transform(lambda x: x.rolling(3, 2).mean())
df["ipca_MA_6"] = df.groupby("id")["ipca_pct"].transform(lambda x: x.rolling(6, 2).mean())
df["ipca_MA_12"] = df.groupby("id")["ipca_pct"].transform(lambda x: x.rolling(12, 2).mean())

# Creating moving averages for caixas_papelao
df["caixas_papelao_MA_2"] = df.groupby("id")["caixas_papelao"].transform(lambda x: x.rolling(2, 2).mean())
df["caixas_papelao_MA_3"] = df.groupby("id")["caixas_papelao"].transform(lambda x: x.rolling(3, 2).mean())
df["caixas_papelao_MA_6"] = df.groupby("id")["caixas_papelao"].transform(lambda x: x.rolling(6, 2).mean())
df["caixas_papelao_MA_12"] = df.groupby("id")["caixas_papelao"].transform(lambda x: x.rolling(12, 2).mean())

# Creating moving averages for caixas_papelao_varpct
df["caixas_papelao_varpct_MA_2"] = df.groupby("id")["caixas_papelao_varpct"].transform(lambda x: x.rolling(2, 2).mean())
df["caixas_papelao_varpct_MA_3"] = df.groupby("id")["caixas_papelao_varpct"].transform(lambda x: x.rolling(3, 2).mean())
df["caixas_papelao_varpct_MA_6"] = df.groupby("id")["caixas_papelao_varpct"].transform(lambda x: x.rolling(6, 2).mean())
df["caixas_papelao_varpct_MA_12"] = df.groupby("id")["caixas_papelao_varpct"].transform(lambda x: x.rolling(12, 2).mean())

# Creating moving averages for demissões
df["demissoes_MA_2"] = df.groupby("id")["demissoes"].transform(lambda x: x.rolling(2, 2).mean())
df["demissoes_MA_3"] = df.groupby("id")["demissoes"].transform(lambda x: x.rolling(3, 2).mean())
df["demissoes_MA_6"] = df.groupby("id")["demissoes"].transform(lambda x: x.rolling(6, 2).mean())
df["demissoes_MA_12"] = df.groupby("id")["demissoes"].transform(lambda x: x.rolling(12, 2).mean())

# Creating moving averages for ind_varejo
df["ind_varejo_MA_2"] = df.groupby("id")["ind_varejo"].transform(lambda x: x.rolling(2, 2).mean())
df["ind_varejo_MA_3"] = df.groupby("id")["ind_varejo"].transform(lambda x: x.rolling(3, 2).mean())
df["ind_varejo_MA_6"] = df.groupby("id")["ind_varejo"].transform(lambda x: x.rolling(6, 2).mean())
df["ind_varejo_MA_12"] = df.groupby("id")["ind_varejo"].transform(lambda x: x.rolling(12, 2).mean())

# Creating moving averages for cambio_dol
df["cambio_dol_MA_2"] = df.groupby("id")["cambio_dol"].transform(lambda x: x.rolling(2, 2).mean())
df["cambio_dol_MA_3"] = df.groupby("id")["cambio_dol"].transform(lambda x: x.rolling(3, 2).mean())
df["cambio_dol_MA_6"] = df.groupby("id")["cambio_dol"].transform(lambda x: x.rolling(6, 2).mean())
df["cambio_dol_MA_12"] = df.groupby("id")["cambio_dol"].transform(lambda x: x.rolling(12, 2).mean())

# Casting types to a lighter format
df = df.astype({"id": "int32", "MCC": "int16", "porte": "int32", "TPVEstimate": "int32",
              "DaysSinceCreation": "int16", "DaysSinceFirstTrans": "int16", "YearCreated": "int16", "MonthCreated": "int8",
              "YearFirstTrans": "int16", "MonthFirstTrans": "int8", "AnoReferencia": "int16", "MesReferencia": "int8"})


# Writing dataframe
df = df.reset_index()
df.to_csv(f"{data_dir}/feat_eng/model_data_v5.csv", index=False)
