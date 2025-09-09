import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv  
API_KEY = st.secrets["TMDB_API_KEY"]
print("Loaded API key:", API_KEY)

API_KEY = "YOUR_TMDB_API_KEY"  
BASE_URL = "https://api.themoviedb.org/3/movie/"
IMG_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, on="title")
    movies = movies[['id','title','overview','genres','keywords','cast','crew','vote_average','release_date','popularity']]
    movies.dropna(inplace=True)

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['id','title','tags','genres','vote_average','release_date','popularity']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

movies, similarity = load_data()

def fetch_poster(movie_id):
    url = f"{BASE_URL}{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get("poster_path", None)
    if poster_path:
        return IMG_URL + poster_path
    else:
        return "https://via.placeholder.com/300x450?text=No+Image"

def recommend(movie, genre_filter=None, min_rating=0, max_year=2100):
    if movie not in movies['title'].values:
        return []

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:20]

    recommendations = []
    for i in movies_list:
        m = movies.iloc[i[0]]
        year = int(m.release_date.split("-")[0]) if m.release_date else 0
        if (genre_filter is None or genre_filter in m.genres) and (m.vote_average >= min_rating) and (year <= max_year):
            recommendations.append({
                "title": m.title,
                "id": m.id,
                "rating": m.vote_average,
                "year": year,
                "poster": fetch_poster(m.id)
            })
        if len(recommendations) >= 5:
            break
    return recommendations

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite one with filters and posters!")

tab1, tab2 = st.tabs(["🍿 Content-Based Filtering", "🤝 Collaborative Filtering"])

    with tab1:
    st.header("Content-Based Recommendations")
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Select a movie:", movie_list)

    genre_filter = st.selectbox("Filter by Genre (optional):", [None] + sorted({g for l in movies.genres for g in l}))
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)
    max_year = st.slider("Maximum Release Year", 1950, 2025, 2025)

    if st.button("Recommend 🎥"):
        recs = recommend(selected_movie, genre_filter, min_rating, max_year)
        if not recs:
            st.warning("No recommendations found. Try adjusting filters.")
        else:
            for rec in recs:
                col1, col2 = st.columns([1,3])
                with col1:
                    st.image(rec['poster'], width=150)
                with col2:
                    st.subheader(f"{rec['title']} ({rec['year']}) ⭐ {rec['rating']}")
                    st.caption(f"Movie ID: {rec['id']}")

with tab2:
    st.header("Collaborative Filtering (Demo)")
    st.info("This could use user rating matrix (Surprise library, ALS, etc.). For now, just a placeholder.")
    st.write("👉 You can extend this to real ratings dataset like MovieLens.")
