import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# Load TMDB API key securely
API_KEY = st.secrets["TMDB_API_KEY"]

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover movies similar to your favorites with filters and posters</p>", unsafe_allow_html=True)

# Dark mode toggle
dark_mode = st.checkbox("Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #0e1117;
            color: white;
        }
        .stButton>button {
            background-color: #1f1f1f;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to fetch popular movies from TMDB
@st.cache_data
def fetch_movies(pages=2):
    all_movies = []
    for page in range(1, pages+1):
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            all_movies.extend(response.json().get("results", []))
        except Exception as e:
            st.error(f"Error fetching data from TMDB: {e}")
            continue
    return all_movies

# Poster URL function
def get_poster(poster_path):
    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path
    return None

# Load movie data
movies_list = fetch_movies()
if not movies_list:
    st.warning("No movies loaded. Check your API key or internet connection.")
    st.stop()

# Create DataFrame and tags
movies_df = pd.DataFrame(movies_list)
movies_df["tags"] = movies_df.apply(lambda row: f"{row.get('title', '')} {row.get('overview', '')} {' '.join(map(str, row.get('genre_ids', [])))}", axis=1)

# Vectorize and calculate similarity
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(movies_df["tags"])
similarity = cosine_similarity(vectors)

# Recommendation function with filters
def recommend(movie_title, min_rating=0, release_year=0):
    if movie_title not in movies_df["title"].values:
        st.warning("Movie not found.")
        return []
    idx = movies_df[movies_df["title"] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
    recs = []
    for i in distances[1:20]:  # get more to allow filtering
        movie = movies_df.iloc[i[0]]
        if movie["vote_average"] >= min_rating and (release_year == 0 or str(release_year) in movie.get("release_date","")):
            recs.append({
                "title": movie["title"],
                "poster": get_poster(movie["poster_path"]),
                "rating": movie["vote_average"],
                "release_date": movie.get("release_date", ""),
                "overview": movie.get("overview", "")
            })
        if len(recs) >=5:
            break
    return recs

# Filters
col1, col2, col3 = st.columns([2,1,1])
with col1:
    movie_selected = st.selectbox("Select a movie:", movies_df["title"].values)
with col2:
    min_rating = st.slider("Minimum rating", 0, 10, 0)
with col3:
    release_year = st.number_input("Release year (optional)", min_value=1900, max_value=2100, value=0)

# Surprise me button
if st.button("Surprise Me"):
    movie_selected = random.choice(movies_df["title"].values)
    st.info(f"Random movie selected: {movie_selected}")

# Show recommendations
if st.button("Show Recommendations") or movie_selected:
    recommendations = recommend(movie_selected, min_rating, release_year)
    if recommendations:
        cols = st.columns(5)
        for i, rec in enumerate(recommendations):
            with cols[i]:
                if rec["poster"]:
                    st.image(rec["poster"], use_column_width=True)
                st.markdown(f"**{rec['title']}**")
                st.markdown(f"Rating: {rec['rating']} | Release: {rec['release_date']}")
                st.markdown(f"*{rec['overview'][:80]}...*")  # short overview
    else:
        st.warning("No recommendations found
