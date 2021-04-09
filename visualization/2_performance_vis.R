library(tidyverse)
library(showtext)
library(Metrics)
library(grid)
library(ggcharts)
library(mdthemes)

font_add(family = "Sharon", regular = "Sharon.ttf")
showtext_auto()
##########################################
rm(list=ls())
options(scipen=999)
##########################################

### Reading dataframes
df_perf = read_csv("Data/Performance/df_performance.csv") %>% 
  select(-1)
df_tpvs = read_csv("Data/Clean/stone_data_cleaned.csv") 

### Getting last five rows for each id
df_tpvs = df_tpvs %>% 
  arrange(id, mes_referencia) %>% 
  group_by(id) %>% 
  slice_tail(n=5) %>% 
  select(-mes_referencia, -TPV_mensal) %>% 
  unique()

### Pivoting dataframes for merge
df_tpv_real = df_perf %>% 
  pivot_longer(starts_with("TPV_t"), values_to="TPV_mensal", names_to="lag") %>% 
  select(id, TPV_mensal, lag)

df_tpv_pred = df_perf %>% 
  pivot_longer(starts_with("TPV_pred_t"), values_to="TPV_pred", names_to="lag") %>% 
  select(TPV_pred, TPV_historic_mean)

### Merging dataframes
df_comp = bind_cols(df_tpv_pred, df_tpv_real) %>% 
  select(id, everything()) %>% 
  mutate(lag=as.numeric(str_remove(lag, "TPV_t")))
rm(df_tpv_real)
rm(df_tpv_pred)

### Transforming df_comp to long format to plot graphs
df_comp_long = df_comp %>% 
  pivot_longer(-c(id, lag), values_to="valor", names_to="tipo")

df_comp_long %>% 
  filter(id==7) %>% 
  ggplot(mapping=aes(x=lag, y=valor, color=tipo))+
  geom_line(size=1.2)+theme_bw()+ xlab("Dia (t+1)") + ylab("TPV (R$)")+
    theme(text=element_text(family="Sharon",size=12, color = "black"))+
    theme(axis.text.x = element_text(colour="black"), axis.text.y = element_text(colour="black"))+
    theme(panel.grid.major = element_line(size=0.1, color="black"))+
    theme(panel.grid.minor = element_line(size=0, color="black"))+
    scale_color_manual(labels=c("Média Histórica","Valor Real", "Modelo"), values=c("#48acf0", "#02111b", "#00a868")) +
    labs(color = "Legenda")+ ggtitle("Performance do Modelo (Client ID = 7)")


### Merging performance dataframe with df_comp
df_complete = inner_join(df_comp, df_tpvs, by="id") %>% 
  select(-lag)

### Grouping by id and getting the mean MAE for each client
df_complete = df_complete %>% 
  group_by(id) %>% 
  summarise(id, MAE_model = mae(TPV_mensal, TPV_pred), MAE_mean=mae(TPV_historic_mean, TPV_mensal), 
    MacroClassificacao, Estado, segmento) %>% unique()



### Mean MAE by MacroClassificacao
perf_macro = df_complete %>% 
  group_by(MacroClassificacao) %>% 
  summarise(mean_mae_model=mean(MAE_model), mean_mae_historic=mean(MAE_mean)) %>% 
  arrange(mean_mae_model) %>% 
  mutate(diff=mean_mae_historic - mean_mae_model)

order_macro = drop_na(perf_macro)
order_macro = order_macro$MacroClassificacao

perf_macro %>% 
  select(-diff) %>% 
  drop_na() %>% 
  pivot_longer(-MacroClassificacao, values_to="Mean_MAE", names_to="tipo") %>% 
  ggplot(mapping=aes(y=Mean_MAE, x=MacroClassificacao,fill=tipo, fill=tipo)) +
  geom_col(color="black") + theme_bw() + xlab("Macro Classificação\n") + ylab("\nMAE Médio (menor é melhor)") +
  theme(text=element_text(family="Sharon",size=12, color = "black")) +
  theme(axis.text.x = element_text(colour="black",size=9), axis.text.y = element_text(colour="black", size=10))+
  theme(panel.grid.major = element_line(size=0))+
  theme(panel.grid.minor = element_line(size=0))+
  scale_fill_manual("Legenda", labels=c("Média Histórica", "Modelo"), values=c("#48acf0", "#00a868"),)+
  ggtitle("MAE Médio do                    e da                   ") +
  scale_x_discrete(limits = order_macro) + 
  annotation_custom(textGrob(expression("Modelo"), gp = gpar(col = "#00a868", fontfamily = "Sharon", cex=1.22),just="left",
                             x = unit(0.41, "npc"), y = unit(1.083, "npc"),hjust=0))+
  annotation_custom(textGrob(expression("Média Histórica"),
                             x = unit(0.9, "npc"), y = unit(1.083, "npc"), gp = gpar(col = "#48acf0", fontfamily = "Sharon", cex=1.20)))+
  theme(legend.position = "none") + coord_flip(clip = 'off')




### Mean MAE by Estado
perf_state = df_complete %>% 
  drop_na() %>% 
  group_by(Estado) %>% 
  summarise(mean_mae_model=mean(MAE_model), mean_mae_historic=mean(MAE_mean)) %>% 
  arrange(mean_mae_model) %>% 
  mutate(diff=mean_mae_historic - mean_mae_model) 

order_est = perf_state$Estado

## Making graph
perf_state %>% 
  select(-diff) %>% 
  drop_na() %>% 
  pivot_longer(-Estado, values_to="Mean_MAE", names_to="tipo") %>% 
  ggplot(mapping=aes(y=Mean_MAE, x=Estado,fill=tipo, fill=tipo)) +
  geom_col(color="black", stat="identity") + theme_bw() + xlab("Estado") + ylab("MAE  Médio  (menor é melhor)") +
  theme(text=element_text(family="Sharon",size=12, color = "black")) +
  theme(axis.text.x = element_text(colour="black",size=7), axis.text.y = element_text(colour="black", size=10))+
  theme(panel.grid.major = element_line(size=0))+
  theme(panel.grid.minor = element_line(size=0))+
  scale_fill_manual("Legenda", labels=c("Média Histórica", "Modelo"), values=c("#48acf0", "#00a868"),)+
  ggtitle("MAE  Médio do                    e da                   ") +
  scale_x_discrete(limits = order_est) +
  annotation_custom(textGrob(expression("Modelo"), gp = gpar(col = "#00a868", fontfamily = "Sharon", cex=1.22),just="left",
                             x = unit(0.275, "npc"), y = unit(1.074, "npc"),hjust=0))+
  annotation_custom(textGrob(expression("Média Histórica"),
                             x = unit(0.638, "npc"), y = unit(1.074, "npc"), gp = gpar(col = "#48acf0", fontfamily = "Sharon", cex=1.20)))+
  theme(legend.position = "none") + coord_cartesian(clip = "off")


 ### Mean MAE by segmento
perf_segmento = df_complete %>% 
  group_by(segmento) %>% 
  summarise(mean_mae_model=mean(MAE_model), mean_mae_historic=mean(MAE_mean)) %>% 
  arrange(mean_mae_model) %>% 
  mutate(diff=mean_mae_historic - mean_mae_model) %>% 
  drop_na()
order_seg = perf_segmento$segmento

perf_segmento %>% 
  select(-diff) %>% 
  pivot_longer(-segmento, values_to="Mean_MAE", names_to="tipo") %>% 
  ggplot(mapping=aes(y=Mean_MAE, x=segmento,fill=tipo, fill=tipo)) +
  geom_col(color="black", stat="identity") + theme_bw() + xlab("Segmento") + ylab("MAE  Médio  (menor é melhor)") +
  theme(text=element_text(family="Sharon",size=12, color = "black")) +
  theme(axis.text.x = element_text(colour="black",size=7), axis.text.y = element_text(colour="black", size=7))+
  theme(panel.grid.major = element_line(size=0))+
  theme(panel.grid.minor = element_line(size=0))+
  scale_fill_manual("Legenda", labels=c("Média Histórica", "Modelo"), values=c("#48acf0", "#00a868"),)+
  ggtitle("MAE Médio do                    e da                   ") +
  scale_x_discrete(limits = order_seg) +
  annotation_custom(textGrob(expression("Modelo"), gp = gpar(col = "#00a868", fontfamily = "Sharon", cex=1.22),just="left",
                             x = unit(0.313, "npc"), y = unit(1.062, "npc"),hjust=0))+
  annotation_custom(textGrob(expression("Média Histórica"),
                             x = unit(0.74, "npc"), y = unit(1.062, "npc"), gp = gpar(col = "#48acf0", fontfamily = "Sharon", cex=1.20)))+
  theme(legend.position = "none") + coord_flip(clip = "off")



### Dispersion graph between TPV_sum and TPV_pred_sum
ggplot(data=df_perf, mapping=aes(y=TPV_sum, x=TPV_pred_sum)) + 
  geom_point(alpha=0.8, color="black") +
  xlim(0, 3000000)+
  ylim(0,2500000)+
  geom_smooth(method="lm", color="#00a868", size=1.3)+
  theme_bw() +
  theme(axis.text.x = element_text(colour="black",size=7), axis.text.y = element_text(colour="black", size=7))+
  theme(panel.grid.major = element_line(size=0))+
  theme(panel.grid.minor = element_line(size=0))+
  theme(text=element_text(family="Sharon",size=12, color = "black"))+
  xlab("Soma do TPV Real") + ylab("Soma do TPV do Modelo")+
  ggtitle("Performance Modelo")

### Dispersion graph between TPV_sum and TPV_mean_sum
ggplot(data=df_perf, mapping=aes(y=TPV_sum, x=TPV_mean_sum)) + 
  geom_point(alpha=0.8, color="black") +
  xlim(0, 3000000)+
  ylim(0,2500000)+
  geom_smooth(method="lm", color="#48acf0", size=1.3)+
  theme_bw()+
  theme(axis.text.x = element_text(colour="black",size=7), axis.text.y = element_text(colour="black", size=7))+
  theme(panel.grid.major = element_line(size=0))+
  theme(panel.grid.minor = element_line(size=0))+
  theme(text=element_text(family="Sharon",size=12, color = "black"))+
  xlab("Soma do TPV Real") + ylab("Soma do TPV da  Média Histórica")+
  ggtitle("Performance Média Histórica")

### comparing both performances
ggplot(data=df_perf, mapping=aes(y=TPV_sum)) + 
  xlim(0, 3000000)+
  ylim(0,2500000)+
  geom_smooth(mapping=aes(x=TPV_pred_sum), method="lm", color="#00a868", size=1.3)+
  geom_smooth(mapping=aes(x=TPV_mean_sum), method="lm", color="#48acf0", size=1.3)+
  theme_bw()+
  theme(axis.text.x = element_text(colour="black",size=7), axis.text.y = element_text(colour="black", size=7))+
  theme(panel.grid.major = element_line(size=0))+
  theme(panel.grid.minor = element_line(size=0))+
  theme(text=element_text(family="Sharon",size=12, color = "black"))+
  xlab("Soma do TPV Real") + ylab("Soma do TPV da Previsto")+
  ggtitle("Performance do                    e Performance da") +
  annotation_custom(textGrob(expression("Modelo"), gp = gpar(col = "#00a868", fontfamily = "Sharon", cex=1.22),just="left",
                             x = unit(0.28, "npc"), y = unit(1.075, "npc"),hjust=0))+
  annotation_custom(textGrob(expression("Média Histórica"),
                             x = unit(0.865, "npc"), y = unit(1.075, "npc"), gp = gpar(col = "#48acf0", fontfamily = "Sharon", cex=1.20)))+
  theme(legend.position = "none") + coord_cartesian(clip = "off")
  

