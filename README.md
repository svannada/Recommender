# Movie Recommender System

## Introduction
This Python script implements a movie recommender system using collaborative filtering and content-based filtering techniques. It provides a Flask-based web service API for users to get personalized movie recommendations based on their preferences and past interactions.

## Collaborative Filtering
### Data Preprocessing
- Loads movie ratings and metadata from CSV files.
- Computes mean ratings and average count of ratings for movies.
- Filters movies based on minimum count and Bayesian average threshold.
- Filters ratings for active users with at least 150 ratings.
- Merges ratings with movies data.
- Generates a user-movie matrix for collaborative filtering.
- Computes cosine similarity between users.

### Functions
1. **recommend_movies_collab(user_id, num_recommendations=5)**:
   - Recommends movies using collaborative filtering for a given user.
   - Returns recommended movies DataFrame.

## Content-Based Filtering
### Data Preprocessing
- Splits genres column into multiple columns.
- Replaces NaN values with blank.
- Extracts release year from movie titles.
- Combines genres, tags, and release year into a final tag.
- Stems final tags for text normalization.
- Applies CountVectorizer to final tags.

### Functions
1. **recommend_movies_content(user_id, num_recommendations=5)**:
   - Recommends movies using content-based filtering for a given user.
   - Returns recommended movies DataFrame.

## Web Service API
- Defines a Flask app with a single endpoint '/recommend' for movie recommendations.
- Accepts parameters like user ID, content slider, user similarity slider, exclude watched, and number of recommendations.
- Integrates collaborative and content-based filtering to generate movie recommendations.
- Returns recommended movies and watched movies with URLs.

## Dependencies
- Flask
- NumPy
- Pandas
- scikit-learn
- nltk
- IMDbPY

## Usage
1. Ensure all dependencies are installed (`pip install -r requirements.txt`).
2. Place CSV files (ratings.csv, movies.csv, links.csv, tags.csv) in the same directory as the script.
3. Run the script (`python movielenshybrid.py`) and access the API endpoint '/recommend' to get movie recommendations.

## Deployed URL
This API is deployed at heroku with the URL - https://data472recommenderapi-dfd873cfc1f2.herokuapp.com/

## Author
Sridhar

## Date
04/18/2024
