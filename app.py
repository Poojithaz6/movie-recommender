import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load TMDB API key securely from Streamlit secrets
API_KEY = st.secrets["TMDB_API_KEY"]

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")

# Function to fetch popular movies from TMDB
def fetch_movies(page=1):
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        st.error("Failed to fetch movies from TMDB")
        return []

# Function to get full poster URL
def get_poster(poster_path):
    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path
    return None

# Fetch first 2 pages of popular movies (for demo; can increase)
movies_list = fetch_movies(page=1) + fetch_movies(page=2)

# Create a DataFrame for processing
movies_df = pd.DataFrame(movies_list)

# Combine title, overview, and genre IDs into a single 'tags' column
def create_tags(row):
    genres = " ".join([str(g) for g in row.get("genre_ids", [])])
    overview = row.get("overview", "")
    title = row.get("title", "")
    return f"{title} {overview} {genres}"

movies_df["tags"] = movies_df.apply(create_tags, axis=1)

# Vectorize tags and calculate similarity
cv = CountVectorizer(stop_words="english")
vectors = cv.fit_transform(movies_df["tags"])
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies_df["title"].values:
        st.warning("Movie not found in dataset")
        return []
    idx = movies_df[movies_df["title"] == movie_title].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)
    recommended = []
    for i in distances[1:6]:  # top 5 recommendations
        recommended.append({
            "title": movies_df.iloc[i[0]]["title"],
            "poster": get_poster(movies_df.iloc[i[0]]["poster_path"]),
            "rating": movies_df.iloc[i[0]]["vote_average"],
            "release_date": movies_df.iloc[i[0]]["release_date"]
        })
    return recommended

# Streamlit UI
movie_selected = st.selectbox(
    "Select a movie for recommendations:",
    movies_df["title"].values
)

if st.button("Show Recommendations"):
    recommendations = recommend(movie_selected)
    cols = st.columns(5)
    for i, rec in enumerate(recommendations):
        with cols[i]:
            if rec["poster"]:
                st.image(rec["poster"], use_column_width=True)
            st.write(rec["title"])
            st.write(f"Rating: {rec['rating']}")
            st.write(f"Release: {rec['release_date']}")
