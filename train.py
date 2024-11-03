import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import pickle
from movie_preprocessing import preprocess_movies_data

def get_preprocessing_params():
    """Define e retorna os parâmetros de pré-processamento."""
    return {
        'remove_cols': [
            'title', 'rating', 'ratingContents', 'originalLanguage', 'writer',
            'soundMix', 'boxOffice', 'distributor', 'audienceScore', 'tomatoMeter'
        ],
        'date_cols': ['releaseDateTheaters', 'releaseDateStreaming'],
        'priority_order': ['releaseDateTheaters', 'releaseDateStreaming'],
        'remove_missing_cols': ['runtime', 'genre', 'director', 'releaseDecade'],
        'rare_label_cols': ['director', 'runtime'],
        'rare_label_tols': [10, 50],
        'vectorize_cols': ['genre', 'director', 'releaseDecade', 'runtime']
    }

def save_pickle(obj, filename):
    """Salva um objeto em um arquivo pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def main():
    # Carrega os dados
    movies_data = pd.read_csv('rotten_tomatoes_movies.csv')
    reviews_data = pd.read_csv('rotten_tomatoes_movie_reviews.csv')

    # Obtém os parâmetros de pré-processamento
    preprocessing_params = get_preprocessing_params()

    # Pré-processa os dados dos filmes
    processed_movies_data = preprocess_movies_data(
        movies_data,
        remove_cols=preprocessing_params['remove_cols'],
        date_cols=preprocessing_params['date_cols'],
        priority_order=preprocessing_params['priority_order'],
        remove_missing_cols=preprocessing_params['remove_missing_cols'],
        rare_label_cols=preprocessing_params['rare_label_cols'],
        rare_label_tols=preprocessing_params['rare_label_tols'],
        vectorize_cols=preprocessing_params['vectorize_cols']
    )

    # Combina os reviews com os dados processados dos filmes
    reviews_movies = pd.merge(
        reviews_data[['id', 'reviewText', 'reviewState']],
        processed_movies_data,
        on='id',
        how='right'
    )
    reviews_movies.dropna(subset=['reviewText', 'id'], inplace=True)
    reviews_movies['reviewState'] = reviews_movies['reviewState'].replace({'fresh': 1, 'rotten': 0})

    # Exporta os dados processados para um arquivo CSV
    reviews_movies.to_csv('processed_movies_data.csv', index=False)

    # Salva as colunas de features e os parâmetros de pré-processamento
    feature_columns = reviews_movies.drop(columns=['id', 'reviewText', 'reviewState']).columns.tolist()
    save_pickle(feature_columns, 'feature_columns.pkl')
    save_pickle(preprocessing_params, 'preprocessing_params.pkl')

    # Prepara os dados para treinamento do modelo
    X_text = reviews_movies['reviewText']
    X_movies = reviews_movies.drop(columns=['id', 'reviewText', 'reviewState'])
    y = reviews_movies['reviewState']

    # Vetoriza o texto dos reviews
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_text_vec = vectorizer.fit_transform(X_text)
    save_pickle(vectorizer, 'vectorizer.pkl')

    # Converte as features dos filmes em matriz esparsa
    X_movies_sparse = sparse.csr_matrix(X_movies.values)

    # Combina as features de texto e filmes
    X_combined = sparse.hstack([X_text_vec, X_movies_sparse])

    # Treina o modelo de Regressão Logística
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_combined, y)
    save_pickle(model, 'trained_model.pkl')

if __name__ == '__main__':
    main()
