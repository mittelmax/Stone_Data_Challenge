library(tidyverse)
library(lubridate)
######################################
rm(list=ls())
options(scipen = 999)
######################################

### Importing dataframe
df = read_csv("Data/Spine/spine_stone_ipea.csv")


### Monthly TPV histogram
ggplot(df, aes(x=TPV_mensal)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white", bins=80)+
  geom_density(alpha=.3, fill="red") +
  xlim(0, 100000) +
  theme_bw() + ylab("Densidade")


### TPV distribution for each Macro Classification
ggplot(drop_na(df), aes(x=TPV_mensal, fill=as.factor(MacroClassificacao))) +
  geom_density(alpha=0.5) +
  xlim(0,50000) +
  theme_bw() +
  guides(fill=guide_legend(title="Macro Classificação")) + 
  xlab("TPV Mensal") + ylab("Densidade")


### Number of clients by state
states = df %>% 
  as_tibble() %>% 
  count(Estado) %>% 
  rename("n_clientes"=n)

ggplot(states, mapping=aes(x=Estado, y=n_clientes, fill=Estado)) +
  geom_col() +
  theme_bw() +
  ylab("Número de clientes") +
  theme(legend.position = "none") 


## Testando relação entre variação percentual da produção de papelão e média do TPV
# Teste para todos os segmentos
tpv_medio = df %>% 
  drop_na %>% 
  group_by(mes_referencia, MacroClassificacao) %>% 
  summarise(med_tpv=mean(TPV_mensal), caixas_papelao_varpct=mean(caixas_papelao_varpct),
            caixas_papelao=mean(caixas_papelao))

cor(tpv_medio$med_tpv,tpv_medio$caixas_papelao_varpct) # Checando correlação 

ggplot(data = tpv_medio, mapping=aes(x=caixas_papelao, y=med_tpv)) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE) +
  facet_wrap(~MacroClassificacao) +
  theme_bw() 


# Teste para setor de eletroeletrônicos
tpv_medio_elet = df %>% 
  drop_na() %>% 
  filter(segmento=="Eletroeletrônicos") %>% 
  group_by(mes_referencia) %>% 
  summarise(med_tpv=mean(TPV_mensal), caixas_papelao_varpct=mean(caixas_papelao_varpct),
            caixas_papelao=mean(caixas_papelao))

cor(tpv_medio_elet$med_tpv,tpv_medio_elet$caixas_papelao_varpct) # Checando correlação 

ggplot(data = tpv_medio_elet, mapping=aes(x=caixas_papelao_varpct, y=med_tpv))+
  geom_point() +
  geom_smooth(method="lm", se=FALSE) +
  theme_bw()
