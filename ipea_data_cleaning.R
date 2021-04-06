library(tidyverse)
####################################
rm(list=ls())
####################################

### Reading dataframes
df_papelao = read_csv("Raw/ipea_papelao.csv")
df_ipca= read_csv("Raw/ipea_ipca.csv")
df_varejo = read_csv("Raw/ipea_varejo.csv", skip=1)
df_demissoes_velho = read_csv("Raw/ipea_demissoes_velho.csv", skip=1)
df_demissoes_novo = read_csv("Raw/ipea_demissoes_novocaged.csv", skip=1)
df_cambio = read_csv("Raw/ipea_cambio.csv")

### Cleaning df_cambio
df_cambio = select(df_cambio, -X3) %>% 
  rename(cambio_dol=2)

### Cleaning df_papelao
df_papelao = select(df_papelao, -X4) %>% 
  rename(caixas_papelao=2, caixas_papelao_varpct=3)

### Cleaning df_ipca
df_ipca = select(df_ipca, -X3) %>% 
  rename(ipca_pct=2)

### Cleaning df_varejo
df_varejo = select(df_varejo, -X49) %>% 
  select(-Código,-Estado) %>% 
  pivot_longer(names_to ="Data", values_to="ind_varejo", cols=-Sigla) %>% 
  mutate(Data = as.numeric(Data))

### Cleaning df_demissoes_velho
df_demissoes_velho = select(df_demissoes_velho, -X37) %>% 
  select(-Código,-Estado) %>% 
  pivot_longer(names_to ="Data", values_to="demissoes", cols=-Sigla)

### Cleaning df_demissoes_velho
df_demissoes_novo = select(df_demissoes_novo, -X16) %>% 
  select(-Código,-Estado) %>% 
  pivot_longer(names_to ="Data", values_to="demissoes", cols=-Sigla)

### Combining demissao_nnovo e demissao_velho
df_demissoes = bind_rows(df_demissoes_novo, df_demissoes_velho)
df_demissoes = mutate(df_demissoes, Data=as.numeric(Data))

### Merging all dataframes
df_ipea_aggr = full_join(df_papelao, df_ipca, by="Data") %>% 
  full_join(df_cambio, by="Data")
df_ipea_state = full_join(df_demissoes, df_varejo)

### Saving dataframes to csv
write_csv(df_ipea_aggr, "Clean/ipea_aggr_cleaned.csv")
write_csv(df_ipea_state, "Clean/ipea_state_cleaned.csv")
