Movie Recommendation System

This project is a Movie Recommendation System built using Python and Streamlit.

It recommends movies similar to a selected movie based on genres, keywords, cast, crew, and overview description.

Users can filter recommendations by genre, minimum rating, and release year.

Movie details and posters are displayed using the TMDB API.

Dataset and Preprocessing:

Uses the TMDB 5000 Movies and Credits dataset from Kaggle.

Missing values are removed for clean processing.

JSON-like columns such as genres, keywords, cast, and crew are parsed into lists.

Overview text is split into words, and all features are combined into a "tags" column.

Tags are vectorized using CountVectorizer, and cosine similarity is calculated for content-based recommendations.

Setup and Usage:

Download the TMDB dataset from Kaggle and place the CSV files in a folder named data inside the repository.

Set your TMDB API key as an environment variable or use Streamlit secrets to fetch movie posters.

Install required Python packages with:

pip install -r requirements.txt


Run the application locally using:

streamlit run app.py


Select a movie from the dropdown menu, apply filters if desired, and see five recommended movies with posters, ratings, and release years.

Tech Stack:

Python, Pandas, NumPy, scikit-learn, requests, Streamlit.

Future Improvements:

Add collaborative filtering using user ratings.

Support multiple genre filters and advanced search.

Deploy the app on Streamlit Cloud for public access.
