import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings and movies data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Group ratings by movieId and calculate count and mean
movie_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])

# Calculate Bayesian average
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()
movie_stats['bayesian_avg'] = (C * m + movie_stats['count'] * movie_stats['mean']) / (C + movie_stats['count'])

# Define minimum count and Bayesian average threshold
min_C = 30
min_m = 3.5

# Filter movies based on thresholds
filtered_movies = movie_stats[(movie_stats['count'] >= min_C) & (movie_stats['bayesian_avg'] >= min_m)]
filtered_movielist = filtered_movies.index

# Get active users with at least 150 ratings
active_users = ratings.groupby('userId')['rating'].count()
active_users = active_users[active_users >= 150]
active_userlist = active_users.index

# Filter ratings for active users and filtered movies
ratings = ratings[ratings['userId'].isin(active_userlist) & ratings['movieId'].isin(filtered_movielist)]

# Merge ratings with movies data
final_ratings = ratings.merge(movies[['movieId', 'title']])

# Pivot table to get a user-movie matrix
pivot_ratings = final_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Compute cosine similarity between users
similarity_scores = cosine_similarity(pivot_ratings)

# Define function to recommend movies for a given user
def recommend_movies_for_user(user_id, top_n=50):
    # Find similar users
    similar_users = np.argsort(similarity_scores[user_id])[::-1][1:]  # Exclude the user itself
    
    # Get movies rated by the user
    user_movies = pivot_ratings.iloc[user_id][pivot_ratings.iloc[user_id] != 0].index
    
    # Initialize dictionary to store movie recommendations
    recommended_movies = {}
    
    # Iterate over similar users
    for similar_user in similar_users:
        # Get movies rated by the similar user
        similar_user_movies = pivot_ratings.iloc[similar_user][pivot_ratings.iloc[similar_user] != 0].index
        
        # Exclude movies already rated by the user
        new_movies = np.setdiff1d(similar_user_movies, user_movies)
        
        # Add new movies to recommendations
        for movie in new_movies:
            if movie in recommended_movies:
                recommended_movies[movie] += 1
            else:
                recommended_movies[movie] = 1
    
    # Sort recommended movies by frequency
    sorted_recommendations = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    
    # Get movie titles
    movie_titles = [movie_id for movie_id, _ in sorted_recommendations[:top_n]]
    
    return movie_titles

# Get movies rated by user 1
user_1_movies = pivot_ratings.iloc[1][pivot_ratings.iloc[1] != 0].index

# Print movies rated by user 1
print("Movies rated by user 1:")
print(user_1_movies)

# Get top 10 recommendations for user 1
recommended_movies_user_1 = recommend_movies_for_user(1)

# Print recommended movies for user 1
print("\nTop 50 recommended movies for user 1:")
for i, movie_id in enumerate(recommended_movies_user_1, 1):
    print(f"{i}. {movie_id}")

# Overall similarity score between user 1 and similar users
overall_similarity_score = similarity_scores[1, :].mean()
print("\nOverall similarity score between user 1 and similar users:", overall_similarity_score)
