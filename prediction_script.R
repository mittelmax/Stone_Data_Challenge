library(tidyverse)

#####################
rm(list=ls())
#####################

df_august = read_csv("Predictions/prediction_august.csv")
df_september = read_csv("Predictions/prediction_september.csv")
df_october = read_csv("Predictions/prediction_october.csv")
df_november = read_csv("Predictions/prediction_november.csv")
df_december = read_csv("Predictions/prediction_december.csv")

# Joining predictions
df_total = bind_cols(df_august, df_september, df_october, df_november, df_december) %>% 
  select(starts_with("TPV"))

# Exporting predicted data
write_csv(df_total, "Predictions/previsao_tpv.csv")
  
