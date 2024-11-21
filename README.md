# Rotten Tomatoes movie review sentiment predictor
Dataset disponível em: https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews?select=rotten_tomatoes_movies.csv

# Pré-requisitos
- Python
- Executar a instalação das dependências em um ambiente virtual com:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Treinando o modelo
1) Descompactar o dataset.7z
2) Rodar o script de treinamento:
```
python train.py
```

# Executando a aplicação
Para iniciar o servidor da aplicação via Streamlit:
```
streamlit run app.py
```