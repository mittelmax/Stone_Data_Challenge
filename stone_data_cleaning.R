library(tidyverse)
library(skimr)
library(stringi)
library(data.table)
library(lubridate)
##############################
rm(list=ls())
options(max.print=100)
##############################

### Importing dataframes
df_cad = read_csv("Raw/cadastrais.csv")
df_data = read_csv("Raw/tpv-mensais-treinamento.csv")
df_siglas = read_csv("Raw/sigla_estados.csv")

#### Merging dataframes
df_cad = distinct(df_cad)
df_total = inner_join(df_cad, df_data, by="id")

##### Data Cleaning
#### Skimminnng through "df_total"
skim(df_total)

### Checking StoneFirstTransactionDate
df_total %>% 
  as_tibble() %>% 
  count(StoneFirstTransactionDate) %>% 
  print(n=20)

## Fixing date formating
df_total$StoneFirstTransactionDate = ymd(df_total$StoneFirstTransactionDate)
skim(df_total$StoneFirstTransactionDate) # -> Almost all column is made by missing values

### Checking mes_referencia
df_total %>% 
  as_tibble() %>% 
  count(mes_referencia) %>% 
  print(n=20)

## Fixing date formating
df_total$mes_referencia = ymd(df_total$mes_referencia)
skim(df_total$mes_referencia) 

### Investigating why there are 61 unique values for "Estado"
unique(df_total$Estado) # both fullname and initials are appearing for each state
                      # Initials are appearing w/ upper and lowercase

## Fixing "Estado" variable
df_siglas = mutate(df_siglas, NOME = paste0(toupper(NOME),"$")) %>% # Converting to lowercase
  mutate(NOME = stri_trans_general(NOME,"Latin-ASCII"))# removing accents

## Creating named list with replacements
siglas = df_siglas$SIGLA
names(siglas) = df_siglas$NOME

df_total = mutate(df_total, Estado = toupper(Estado)) %>% # Converting to uppercase
  mutate(Estado = stri_trans_general(Estado,"Latin-ASCII")) # removing accents

## Replacing strings
df_total$Estado = str_replace_all(df_total$Estado, siglas)
unique(df_total$Estado)

df_total %>% 
  as_tibble %>% 
  count(porte)

### Checking MCC
df_total %>% 
  as_tibble() %>% 
  count(MCC) %>% 
  print(n=300)

### Checking MacroClassificação
df_total %>% 
  as_tibble() %>% 
  count(MacroClassificacao)

### Checking Segmento
df_total %>% 
  as_tibble() %>% 
  count(segmento) %>% 
  print(n=40)

### Checking Subsegmento
df_total %>% 
  as_tibble() %>% 
  count(sub_segmento) %>% 
  print(n=100)

### Checking Persona
df_total %>% 
  as_tibble() %>% 
  count(persona) %>% 
  print(n=100)

### Checking Porte
df_total %>% 
  as_tibble() %>% 
  count(porte) %>% 
  print(n=100)

## Changing porte datatype to number
df_total = mutate(df_total, porte=str_replace_all(porte, "2.5k", "2500"))
df_total = mutate(df_total, porte=str_replace_all(porte, "k", "000"))
df_total = mutate(df_total, porte=str_replace_all(porte, "\\+", ""))
df_total = mutate(df_total, porte=str_replace_all(porte, "-", "+"))
eval_parse = function (x) {eval(parse(text = x))}
df_total$porte = sapply(df_total$porte, eval_parse)/2

### Checking TPVEstimate
ggplot(data=df_total, mapping=aes(x=TPVEstimate)) +
  geom_histogram(bins=20) +
  xlim(0, 100000) +
  theme_bw()

### Checking tipo_documento
df_total %>% 
  as_tibble() %>% 
  count(tipo_documento) %>% 
  print(n=100)

### The dataframe has different entries for the same id's 
distinct_values = df %>% 
  as_tibble() %>% 
  group_by(id) %>% 
  summarise(n_mcc = n_distinct(MCC), n_macro=n_distinct(MacroClassificacao),
            n_segmento=n_distinct(segmento), n_subsegmento=n_distinct(sub_segmento),
            n_persona=n_distinct(persona), n_porte=n_distinct(porte),
            n_tpvestimate = n_distinct(TPVEstimate), n_documento=n_distinct(tipo_documento),
            n_estado=n_distinct(Estado))

## These ids have different caracteristics
ids_errados = distinct_values %>% 
  filter(n_mcc>1 | n_macro>1 | n_segmento>1 | n_subsegmento>1 | n_persona>1 |
           n_porte>1 | n_tpvestimate>1 | n_documento>1 | n_estado>1) 

## Defining  a strategy to keep the most recent entries for each id
linhas_certas = tibble()
contador = 1
for (id_problematico in ids_errados$id) 
{
  df_i = df %>% 
    as_tibble() %>% 
    filter(id==id_problematico) %>% 
    arrange(Estado, desc(TPVEstimate), desc(porte)) ## Prioritizing entries with a valid state variable
                                                    ## and with the highest TPV estimate and size
  df_i = slice_head(df_i, n=n_distinct(df_i$mes_referencia))
  
  linhas_certas = bind_rows(linhas_certas, df_i)
  print(contador)
  contador = contador + 1
}


## Eliminating wrong id's from dataframe
df_correct = df %>% 
  as_tibble() %>% 
  filter(!(id %in% ids_errados$id))
df_correct = bind_rows(df_correct, linhas_certas) %>% 
  arrange(id, mes_referencia)

#### Saving cleaned dataframe
write_csv(df_correct, "Clean/stone_data_cleaned.csv")
