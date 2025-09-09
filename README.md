# Movie Recommendation System

This project is a content-based movie recommendation system that suggests movies to users based on thei>

## Features

- Fetches live movie data from TMDB API (no local CSV required)
- Content-based recommendations using movie title, overview, and genre
- Minimum rating filter to refine results
- Optional release year filter to narrow suggestions
- Dark/Light mode toggle for better UI experience
- “Surprise Me” button for random movie selection
- Displays movie posters, rating, release year, and a short overview
- Handles errors gracefully when data is missing or API requests fail

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set your TMDB API key as a Streamlit secret. Create a file at .streamlit/secrets.toml with the followin>

toml
Copy code
TMDB_API_KEY = "your_api_key_here"
Run the application:

bash
Copy code
streamlit run app.py

         
