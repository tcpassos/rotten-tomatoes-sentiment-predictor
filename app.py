import streamlit as st
import pandas as pd
from scipy import sparse
import pickle
import matplotlib.pyplot as plt
import base64

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
def load_reviews_data():
    reviews_data = pd.read_csv('rotten_tomatoes_movie_reviews.csv')
    reviews_data.set_index('id', inplace=True)
    return reviews_data

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

def display_pie_chart(reviews_data, movie_id):
    movie_reviews = reviews_data.loc[movie_id]
    sentiment_counts = movie_reviews['reviewState'].value_counts()
    sizes = [sentiment_counts.get('fresh', 0), sentiment_counts.get('rotten', 0)]
    colors = ['#f23535','#668132']
    explode = (0.1, 0)

    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 14})
    ax1.axis('equal')

    fig1.patch.set_alpha(0.0) 
    st.pyplot(fig1)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    st.title("Análise de Sentimento de Reviews de Filmes")
    st.write("Selecione um filme e insira seu review para prever o sentimento.")

    # Carrega o modelo, o vectorizer e as colunas de features
    model, vectorizer, feature_columns = load_model_and_vectorizer()

    # Carrega os dados dos filmes
    movies_data = load_data()

    # Carrega os dados dos reviews
    reviews_data = load_reviews_data()

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

    # Exibe as informações do filme
    st.subheader(f"Informações do Filme: {movie_info.get('title', 'Título desconhecido')}")
    
    # Cria duas colunas
    col1, col2 = st.columns([3, 1])

    with col1:
        # Exibe informações adicionais sobre o filme imediatamente após a seleção
        st.write(f"**Gênero:** {movie_info.get('genre', 'Desconhecido')}")
        st.write(f"**Diretor:** {movie_info.get('director', 'Desconhecido')}")
        st.write(f"**Duração:** {movie_info.get('runtimeMinutes', 'Desconhecido')} minutos")
        release_date = pd.to_datetime(movie_info.get('releaseDateTheaters'), errors='coerce')
        release_date = release_date.strftime('%d/%m/%Y') if not pd.isna(release_date) else 'Desconhecida'
        st.write(f"**Data de Lançamento:** {release_date}")

    with col2:
        # Exibe o gráfico de pizza com os reviews do filme
        display_pie_chart(reviews_data, movie_id)

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
            fresh_img_base64 = get_base64_image("Fresh.png")
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{fresh_img_base64}" width="50"/>
                    <span style="color: green; font-weight: 500; margin-left: 10px;">O review é previsto como <strong>POSITIVO</strong>.</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            rotten_img_base64 = get_base64_image("Rotten.png")
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{rotten_img_base64}" width="50"/>
                    <span style="color: red; font-weight: 500; margin-left: 10px;">O review é previsto como <strong>NEGATIVO</strong>.</span>
                </div>
                """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()