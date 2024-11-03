import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle

@st.cache_data
def load_data():
    # Carrega os dados processados dos filmes
    processed_movies_data = pd.read_csv('processed_movies_data.csv')
    
    # Carrega os dados originais dos filmes e remove duplicatas
    original_movies_data = pd.read_csv('rotten_tomatoes_movies.csv')
    original_movies_data = original_movies_data.drop_duplicates(subset='id')
    original_movies_data = original_movies_data[['id', 'title', 'genre', 'director', 'runtimeMinutes', 'releaseDateTheaters']]
    
    # Mescla os dados originais com os dados processados
    movies_data = pd.merge(
        processed_movies_data,
        original_movies_data,
        on='id',
        how='left'
    )
    
    return movies_data

@st.cache_data
def load_model_and_vectorizer():
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('feature_columns.pkl', 'rb') as file:
        feature_columns = pickle.load(file)
    return model, vectorizer, feature_columns

def get_movie_features(movie_id, movies_data, feature_columns): 
    movie = movies_data[movies_data['id'] == movie_id]
    if movie.empty:
        return None, None
    movie_features = movie[feature_columns].iloc[0]
    movie_info = movie[['title', 'genre', 'director', 'runtimeMinutes', 'releaseDateTheaters']].iloc[0]
    return movie_features, movie_info

def main():
    st.title("Análise de Sentimento de Reviews de Filmes")
    st.write("Selecione um filme e insira seu review para prever o sentimento.")

    # Carrega o modelo, o vectorizer e as colunas de features
    model, vectorizer, feature_columns = load_model_and_vectorizer()

    # Carrega os dados dos filmes
    movies_data = load_data()

    # Cria um dicionário de IDs e títulos dos filmes processados
    movie_options = movies_data[['id', 'title']]
    movie_dict = pd.Series(movie_options.title.values, index=movie_options.id).to_dict()

    # Lista de IDs e títulos para o selectbox
    movie_ids = list(movie_dict.keys())
    movie_titles = [f"{movie_dict[mid]} (ID: {mid})" for mid in movie_ids]

    # Input para seleção do filme
    selected_movie = st.selectbox("Selecione o filme:", movie_titles, index=None)
    # Corrige a extração do movie_id
    if selected_movie is not None:
        movie_id = selected_movie.split(" (ID: ")[-1].rstrip(")")
        movie_id = movie_id.strip()
    else:
        movie_id = None

    # Obtém as features e informações do filme selecionado
    movie_features, movie_info = get_movie_features(movie_id, movies_data, feature_columns)
    if movie_features is None:
        st.error("ID do filme não encontrado nos dados processados.")
        return

    # Exibe informações adicionais sobre o filme imediatamente após a seleção
    st.subheader(f"Informações do Filme: {movie_info.get('title', 'Título desconhecido')}")
    st.write(f"**Gênero:** {movie_info.get('genre', 'Desconhecido')}")
    st.write(f"**Diretor:** {movie_info.get('director', 'Desconhecido')}")
    st.write(f"**Duração:** {movie_info.get('runtimeMinutes', 'Desconhecido')} minutos")
    st.write(f"**Data de Lançamento:** {movie_info.get('releaseDateTheaters', 'Desconhecida')}")

    # Input para o texto do review
    review_text = st.text_area("Digite o seu review:", height=200)

    # Botão para prever o sentimento
    if st.button("Prever Sentimento"):
        # Converte as features do filme em matriz esparsa
        X_movies_sparse = sparse.csr_matrix(movie_features.values.reshape(1, -1))
        # Vetoriza o texto do review
        X_text_vec = vectorizer.transform([review_text])
        # Combina as features de texto e filme
        X_combined = sparse.hstack([X_text_vec, X_movies_sparse])
        # Faz a previsão
        prediction = model.predict(X_combined)
        # Interpreta a previsão
        if prediction[0] == 1:
            st.success("O review é previsto como **POSITIVO**.")
        else:
            st.warning("O review é previsto como **NEGATIVO**.")

if __name__ == '__main__':
    main()
