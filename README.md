# Análise de Sentimento

Este projeto implementa um modelo de análise de sentimento usando scikit-learn para classificar textos em português como positivos ou negativos.

## Requisitos

- Python 3.x
- pandas
- scikit-learn
- nltk

Instale as dependências com:
```
pip install pandas scikit-learn nltk
```

## Como Usar

1. Baixe o dataset `B2W-Reviews01.csv` do [repositório original](https://github.com/americanas-tech/b2w-reviews01) e coloque na pasta do projeto.
2. Execute o script:
   ```
   python main.py
   ```
   Isso treinará o modelo e classificará o sentimento do texto em `text.txt`.

## Estrutura do Projeto

- `main.py`: Código principal.
- `text.txt`: Texto de exemplo para classificação.
- `B2W-Reviews01.csv`: Dataset de treinamento (baixe separadamente).

## Funcionalidades

- Carregamento de dados do CSV.
- Treinamento de modelo com TF-IDF e Regressão Logística.
- Classificação de sentimento de um texto.