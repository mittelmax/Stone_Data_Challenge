# Stone Data Challenge - Previsão de TPV

## `Introdução`

Gostaria de aproveitar esse espaço para me apresentar brevemente. Meu nome é Max e estou atualmente cursando o quinto semestre de economia no Insper. Me interesso muitíssimo pela área de ciência de dados e tenho intenção de buscar uma carreira nessa área. Por esse motivo, busco ampliar meus conhecimentos na área por meio da participação em organizações estudantis, estágios de férias e competições do Kaggle.

Para encerrar, gostaria de dizer que foi uma experiência muito prazerosa participar deste desafio. Encontrei diversos problemas que me fizeram sair de minha zona de conforto. Espero também que algumas das ideias que surgiram deste projeto sejam interessantes para vocês :)

## `Estrutura do Diretório`

Aqui está um resumo da estrutura desse diretório. Aloquei todos arquivos da forma que me pareceu mais coerente.

* `stone_data_challenge/`
  * `stone_data_challenge/`
    * __init__.py
    * config.py
  * `data/`
    * `raw/`
    * `clean/`
    * `feat_eng/`
    * `model/`
    * `predictions/`
  * `scripts/`
    * 1_stone_data_cleaning.R
    * 2_ipea_data_cleaning.R
    * 3_join_stone_ipea.R
    * 4_feat_engineering.py
    * 5_model_lightgbm_t1.py
    * 6_model_lighgbm_t2.py
    * 7_model_lightgbm_t3.py
    * 8_model_lightgbm_t4.py
    * 9_model_lightgbm_t5.py
    * 10_model_historic_mean.py
    * 11_performance.py
    * 12_prediction_join.py
  * `visualization/`
    * 1_dataviz.R
    * 2_performance_vis.R
    * 3_markdown_cleaning.Rmd
    * `fonts/`
    * `html/`

Agora, vamos analisar cada uma das partes em ordem:

## `stone_data_challenge`
* `stone_data_challenge/`
  * `stone_data_challenge/`
    * __init__.py
    * config.py

Essa pasta contém as configurações do projeto. Em essência, é um pacote que contém o path para a pasta "data". Esse pacote é chamado nos scripts de Python para utilizar o path mencionado.

## `data`
* `data/`
  * `raw/`
  * `clean/`
  * `feat_eng/`
  * `model/`
    * `parameters/`
    * `performance/`
  * `predictions/`
  
Essa pasta contém todas as bases de dados utilizadas, dados sobre a performance do modelo, e a previsão final para ser contabilizada para o desafio:

* `raw` - Contém todas as bases de dados necessárias para reproduzir meus resultados. Enviei o conteúdo dessa pasta para vocês por email para que vocês consigam reproduzir meus resultados.
* `clean` - Contém as bases de dados depois de serem limpas com os scripts em R.
* `feat_eng` - Contém a base de dados depois do processo de feature engineering (é a base que foi utilizada pelo modelo final). 
* `model` - Contém tanto a melhor seleção de parâmetros para os modelos, quanto as métricas de performance no test set.
* `predictions` - Contém as previsões finais para o desafio.

## `scripts`
* `scripts/`
  * 1_stone_data_cleaning.R
  * 2_ipea_data_cleaning.R
  * 3_join_stone_ipea.R
  * 4_feat_engineering.py
  * 5_model_lightgbm_t1.py
  * 6_model_lighgbm_t2.py
  * 7_model_lightgbm_t3.py
  * 8_model_lightgbm_t4.py
  * 9_model_lightgbm_t5.py
  * 10_model_historic_mean.py
  * 11_performance.py
  * 12_prediction_join.py

Antes de qualquer coisa, quero frisar que os scripts estão enumerados na ordem em que devem ser executados para reproduzir meus resultados. Além disso, é possível observar alguns scripts de R no meio de todos esses scripts. Tal motivo veio de uma preferência minha em utilizar o R para a limpeza de dados quando o tamanho das bases não é excessivamente grande. O nível de produtividade e liberdade que o tidyverse fornece é algo sem paralelos, em minha opinião.

* `Scripts 1 a 3` - São os scripts de limpeza e join das bases de dados utilizadas.
* `Script 4` - É o script de feature enngineering.
* `Scripts 5 a 9` - São os scripts dos modelos utilizados para a previsão.
* `Script 10` - É o script para a performance da média histórica no test set.
* `Script 11` - É o script que une todos os resultados de performance do modelo.
* `Script 12` - É o script que une todos as previsões finais do modelo.

## `visualization`
* `visualization/`
  * 1_dataviz.R
  * 2_performance_vis.R
  * 3_markdown_cleaning.Rmd
  * `fonts/`
  * `html/`

Essa pasta contém todos os scripts utilizados para a construção de gráficos, visualizações sobre as variáveis e performance do modelo. Além disso, também está contido nesta pasta o arquivo final de visualização (em html) que foi construído através de um R Markdown.

## `Dúvidas sobre execução`
Peço encarecidamente que enviem qualquer dúvida em relação à execução do código para meu email `maxmitteldorf@gmail.com`
