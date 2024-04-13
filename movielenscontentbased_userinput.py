import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Load ratings, movies, and tags data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

# Splitting a single genre column into multiple columns
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

movie_tags = tags.groupby('movieId')['tag'].agg(list).reset_index()
movies_final = movies.merge(movie_tags, on='movieId', how='left')

# Replace NaN with blank
movies_final['tag'] = movies_final['tag'].fillna('')
movies_final['genres'] = movies_final['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies_final['tag'] = movies_final['tag'].apply(lambda x: [i.replace(" ", "") for i in x])

# Extract the release year from the movie title
pattern = r'\((\d{4})\)$'
movies_final['release-year'] = movies_final['title'].str.extract(pattern)

# Convert release year to array format
movies_final['release-year'] = movies_final['release-year'].apply(lambda year: [year])

# Create final tag combining genres, tags, and release year
movies_final['final_tag'] = movies_final['genres'] + movies_final['tag'] + movies_final['release-year']
movies_df = movies_final[['movieId', 'title', 'final_tag']]

# Convert the 'final_tag' column from list format to string format
movies_df['final_tag'] = movies_df['final_tag'].apply(lambda x: ' '.join(map(str, x)))

# Stemming function
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Stemming final tags
movies_df['final_tag'] = movies_df['final_tag'].apply(stem)

# CountVectorizer
cv = CountVectorizer(max_features=1600, stop_words='english')
vectors = cv.fit_transform(movies_df['final_tag']).toarray()

def recommend_movies_for_user(user_id, top_n=50):
    # Find movies rated by the user
    user_movies = ratings[ratings['userId'] == user_id].merge(movies_df, on='movieId', how='left')
    
    user_vectors = cv.transform(user_movies['final_tag']).toarray()
    
    # Compute cosine similarity between user's movies and all movies
    user_similarity  = cosine_similarity(user_vectors, vectors)

    # Get average similarity score for each movie rated by the user
    avg_similarity = np.mean(user_similarity, axis=0)
    
    # Get indices of movies sorted by similarity score
    sorted_indices = np.argsort(avg_similarity)[::-1]
    
    # Get movie titles based on sorted indices
    recommended_movies = [movies_df.iloc[idx]['title'] for idx in sorted_indices[:top_n]]
    
    return recommended_movies, avg_similarity

# Get recommendations for a user
user_id = 1
recommended_movies, overall_similarity_score = recommend_movies_for_user(user_id)
print(f"Movies rated by user {user_id}:")
user_ratings = ratings[ratings['userId'] == user_id]
user_movie_titles = movies[movies['movieId'].isin(user_ratings['movieId'])]['title'].tolist()
for i, movie_title in enumerate(user_movie_titles, 1):
    print(f"{i}. {movie_title}")

print(f"\nTop 50 recommended movies for user {user_id}:")
for i, movie_title in enumerate(recommended_movies, 1):
    print(f"{i}. {movie_title}")

print(f"\nOverall similarity score for user {user_id}: {np.mean(overall_similarity_score)}")