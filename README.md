# Conferencia D02

Este repositorio contem os arquivos necessarios para reproduzir o processamento do caso D02.

## Execucao

No PowerShell, a partir da raiz do repositorio:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN.ps1
```

Na primeira execucao, o ambiente virtual local e criado automaticamente e as dependencias sao instaladas a partir de `requirements.txt`.

## Estrutura

- `02_model_input/modelo01/modelo01_d02_survival_dataset.csv`
- `scripts/python/`
- `docs/`

## Saidas geradas

A execucao gera os artefatos numericos em `02_model_input/modelo02/`, as figuras em `outputs/relatorio_graficos_modelo02/` e o relatorio tecnico em `docs/RELATORIO_TECNICO_D02.md`.
