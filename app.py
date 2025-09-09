import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

API_KEY = st.secrets["TMDB_API_KEY"]

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover movies similar to your favorites!</p>", unsafe_allow_html=True)

dark_mode = st.checkbox("Dark Mode")
if dark_mode:
    st.markdown("""
    <style>
    .reportview-container {background-color: #0e1117; color: white;}
    .stButton>button {background-color: #1f1f1f; color: white;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def fetch_movies(pages=2):
    all_movies = []
    for page in range(1, pages + 1):
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            all_movies.extend(response.json().get("results", []))
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from TMDB: {e}")
    if not all_movies:
        st.stop()
    return all_movies

movies_list = fetch_movies()
movies_df = pd.DataFrame(movies_list)
movies_df["tags"] = movies_df.apply(lambda row: f"{row.get('title','')} {row.get('overview','')} {' '.join(map(str,row.get('genre_ids',[])))}", axis=1)

cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(movies_df["tags"])
similarity = cosine_similarity(vectors)

def get_poster(poster_path):
    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path
    return "https://via.placeholder.com/300x450?text=No+Image"

def recommend(movie_title, min_rating=0, release_year=0):
    if movie_title not in movies_df["title"].values:
        st.warning("Movie not found in dataset.")
        return []
    idx = movies_df[movies_df["title"] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
    recs = []
    for i in distances[1:50]:
        movie = movies_df.iloc[i[0]]
        rating = movie.get("vote_average", 0)
        release = movie.get("release_date", "")
        if rating >= min_rating and (release_year == 0 or str(release_year) in release):
            recs.append({
                "title": movie.get("title", "Unknown"),
                "poster": get_poster(movie.get("poster_path")),
                "rating": rating,
                "release_date": release,
                "overview": movie.get("overview", "No description available")
            })
        if len(recs) >= 5:
            break
    if not recs:
        st.info("No recommendations found with current filters.")
    return recs

st.sidebar.header("Filters")
min_rating = st.sidebar.slider("Minimum Rating", 0, 10, 0)
release_year = st.sidebar.number_input("Release Year (optional)", min_value=0, max_value=2100, value=0)
col1, col2 = st.columns([3,1])
with col1:
    movie_selected = st.selectbox("Select a movie:", movies_df["title"].values)
with col2:
    if st.button("Surprise Me"):
        movie_selected = random.choice(movies_df["title"].values)
        st.info(f"Random movie selected: {movie_selected}")

if st.button("Show Recommendations") or movie_selected:
    try:
        recommendations = recommend(movie_selected, min_rating, release_year)
        if recommendations:
            cols = st.columns(5)
            for i, rec in enumerate(recommendations):
                with cols[i]:
                    st.image(rec["poster"], use_column_width=True)
                    st.markdown(f"**{rec['title']}**")
                    st.markdown(f"Rating: {rec['rating']} | Release: {rec['release_date']}")
                    st.markdown(f"*{rec['overview'][:80]}...*")
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
