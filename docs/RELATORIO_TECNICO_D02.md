# RELATÓRIO TÉCNICO - D02

## Regras do Modelo02
- Regra mínima: substituição completa só encerra episódio se ocorrer com pelo menos 8 dias desde a última substituição válida.
- Eventos curtos com 6 ou 7 dias permanecem registrados no histórico, mas não reiniciam o ciclo analítico.
- Remoção para modelagem: episódios válidos classificados como outlier alto por duração acima de média + 2 desvios padrão são retirados do ajuste estatístico principal.
- Melhor prática: primeira e segunda melhores práticas são selecionadas apenas entre episódios válidos não classificados como outlier.

## Resumo numérico
- Base original D02: 34 episódios, 33 eventos observados, 1 censurado(s), média 222,85 dias e mediana 199 dias.
- Base validada com regra mínima: 29 episódios, 28 eventos observados, 1 censurado(s), média 262,64 dias e mediana 240 dias.
- Base de modelagem do Modelo02: 27 episódios, 26 eventos observados, 1 censurado(s), média 226,54 dias e mediana 220 dias.

## Limites de classificação
- Média dos episódios válidos: 262,64 dias.
- Desvio padrão: 197,99 dias.
- Limite inferior da faixa típica: 64,66 dias.
- Limite superior da faixa típica: 460,63 dias.
- Limiar de outlier alto: 658,62 dias.

## Modelo de Cox
- Cox A status: OK.
- Cox A C-index: 0,942197.
- Cox B status: SKIPPED.

## Comparação entre modelos
- Modelo escolhido: CoxPH.
- Justificativa: Sem alternativa com ganho consistente; manter CoxPH por interpretabilidade e robustez..

## Arquivos principais
- `02_model_input/modelo02/modelo02_d02_survival_dataset_validado.csv`
- `02_model_input/modelo02/modelo02_d02_survival_dataset.csv`
- `02_model_input/modelo02/modelo02_d02_episode_classificacao.csv`
- `outputs/relatorio_graficos_modelo02/`
