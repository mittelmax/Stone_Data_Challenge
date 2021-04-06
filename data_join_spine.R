library(tidyverse)
library(lubridate)
library(zoo)
###################################
rm(list=ls())
###################################

### Reading dataframes
df_stone = read_csv("Clean/stone_data_cleaned.csv")
df_ipea_aggr = read_csv("Clean/ipea_aggr_cleaned.csv")
df_ipea_state = read_csv("Clean/ipea_state_cleaned.csv")

### Fixing date formatting for merge
df_ipea_aggr$Data = str_replace_all(df_ipea_aggr$Data, "\\.1$", ".10")
df_ipea_aggr$Data = format_ISO8601(ymd(paste0(df_ipea_aggr$Data,".27")),precision = "ym")
df_ipea_aggr = rename(df_ipea_aggr, mes=Data)

df_ipea_state$Data = str_replace_all(df_ipea_state$Data, "\\.1$", ".10")
df_ipea_state$Data = format_ISO8601(ymd(paste0(df_ipea_state$Data,".27")),precision = "ym")
df_ipea_state = rename(df_ipea_state, mes=Data, Estado=Sigla)

df_stone = mutate(df_stone, mes = format_ISO8601(mes_referencia, precision = "ym"))

### Merging dataframes
df_total = left_join(df_stone, df_ipea_aggr, by=c("mes"))
df_total = left_join(df_total, df_ipea_state)

### Saving dataframe
write_csv(df_total, "Spine/spine_stone_ipea.csv")