from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS


app = Flask(__name__)

# List of allowed origins
allowed_origins = [
    "http://example1.com",
    "http://localhost:3000"
]

# Configure CORS with allowed origins list
cors = CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

@app.route('/')
def home():
    return 'Ethical Movie Recommender API'

# Load ratings and movies data
ref_ratings = pd.read_csv('ml-latest-small/ratings.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
# Load movie links data
movie_links = pd.read_csv('ml-latest-small/links_with_url.csv')
# Load tags data
tags = pd.read_csv('ml-latest-small/tags.csv')

# Collaborative Filtering Pre processing logic

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


# Content Based Filtering Pre processing logic

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

# Create pre link and post link dfs to merge the url links
movies_prelink_df = movies_final[['movieId', 'title', 'final_tag']]
movies_postlink_df = movies_prelink_df.merge(movie_links, on='movieId')

# Final data frame to be used in the recommendations
movies_df = movies_postlink_df[['movieId', 'title', 'final_tag', 'cover_url']]

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


def recommend_movies_collab(user_id, watched_movies, num_recommendations=5):
    if(num_recommendations) > 0:
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
                    recommended_movies[movie] += similarity_scores[user_id][similar_user]  # Use similarity score here
                else:
                    recommended_movies[movie] = similarity_scores[user_id][similar_user]  # Use similarity score here
        
        # Convert the watched movie titles to a set for faster lookup
        watched_movie_titles = set(watched_movies['title'])

        # Filter out watched movies from recommended movies
        filtered_recommendations = {title: score for title, score in recommended_movies.items() if title not in watched_movie_titles}

        # Sort filtered recommended movies by score
        sorted_recommendations = sorted(filtered_recommendations.items(), key=lambda x: x[1], reverse=True)

        # Get movie titles from sorted recommendations
        movie_titles = [title for title, _ in sorted_recommendations[:num_recommendations]]

        # Filter movies from movies_df based on titles
        recommended_movies = movies_df[movies_df['title'].isin(movie_titles)]
    else:
        recommended_movies  = pd.DataFrame()
    return recommended_movies

def recommend_movies_content(user_id, watched_movies, num_recommendations=5):
    if(num_recommendations) > 0:
        # Find movies rated by the user
        user_movies = ratings[ratings['userId'] == user_id].merge(movies_df, on='movieId', how='left')
        
        user_vectors = cv.transform(user_movies['final_tag']).toarray()
        
        # Compute cosine similarity between user's movies and all movies
        user_similarity  = cosine_similarity(user_vectors, vectors)

        # Get average similarity score for each movie rated by the user
        avg_similarity = np.mean(user_similarity, axis=0)
        
        # Get indices of movies sorted by similarity score
        sorted_indices = np.argsort(avg_similarity)[::-1]

        # Filter out watched movies from sorted indices
        filtered_indices = [index for index in sorted_indices if movies_df.iloc[index]['movieId'] not in watched_movies['movieId']]

        # Get movie titles from filtered indices
        movie_titles = movies_df.iloc[filtered_indices[:num_recommendations]]['title']

        # Use movie_titles to create a boolean mask
        mask = movies_df['title'].isin(movie_titles)

        # Apply the mask to movies_df to get recommended movies
        recommended_movies = movies_df[mask]
    else:
        recommended_movies = pd.DataFrame()

    return recommended_movies


@app.route('/api/recommend', methods=['GET'])
def recommend_movies():
    user_id = int(request.args.get('userId'))
    content_slider = float(request.args.get('contentSlider'))
    user_similarity_slider = float(request.args.get('userSimilaritySlider'))
    num_recommendations = int(request.args.get('numRecommendations'))

    num_recommendations_collab = int(user_similarity_slider * num_recommendations)
    num_recommendations_content = int(content_slider * num_recommendations)

    user_ratings =  ratings[ratings['userId'] == user_id]['movieId']
    # Use movie_titles to create a boolean mask
    watch_mask = movies_df['movieId'].isin(user_ratings)

    # Apply the mask to movies_df to get recommended movies
    watched_movies = movies_df[watch_mask]

    recommended_movies_collab = recommend_movies_collab(user_id, watched_movies, num_recommendations_collab)
    recommended_movies_content = recommend_movies_content(user_id, watched_movies, num_recommendations_content)

    recommended_movies = pd.concat([recommended_movies_collab, recommended_movies_content], ignore_index=True)

    # Convert DataFrames to dictionaries
    recommended_movies_dict = recommended_movies.to_dict(orient='records')
    watched_movies_dict = watched_movies.to_dict(orient='records')

    # Return as JSON response
    return jsonify({
        'recommended_movies': recommended_movies_dict,
        'watched_movies': watched_movies_dict
    })

@app.route('/api/popularusers', methods=['GET'])
def popular_users():
    # Get popular users with at least 500 ratings
    popular_users = ref_ratings.groupby('userId')['rating'].count()
    popular_users = popular_users[popular_users >= 500]
    popular_userlist = popular_users.index.tolist()
    # Return as JSON response
    return jsonify({
        'popular_users': popular_userlist
    })


if __name__ == '__main__':
    app.run()
