import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

movie_stats = ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
C = movie_stats['count'].mean()
m = movie_stats['mean'].mean()

movie_stats['bayesian_avg'] = (C * m + movie_stats['count'] * movie_stats['mean']) / (C + movie_stats['count'])

min_C = 30
min_m = 3.5
# Filter out movies with count below min_C and Bayesian average below min_m
filtered_movies = movie_stats[(movie_stats['count'] >= min_C) & (movie_stats['bayesian_avg'] >= min_m)]

filtered_movielist = filtered_movies.index

# Lets get the top interactive users as well 
active_users = ratings.groupby('userId')['rating'].agg(['count'])
active_users = active_users[active_users['count'] >= 150]
active_userlist = active_users.index
ratings = ratings[ratings['userId'].isin(active_userlist)]
final_ratings = ratings[ratings['movieId'].isin(filtered_movielist)]
final_ratings = final_ratings.merge(movies[['movieId', 'title']])
final_ratings.drop_duplicates()
pivot_ratings = final_ratings.pivot_table(index = 'title',columns = 'userId', values = 'rating')
pivot_ratings.fillna(0,inplace=True)

similarity_scores = cosine_similarity(pivot_ratings)
def recommend(movie_name):
    # index fetch
    index = np.where(pivot_ratings.index == movie_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key = lambda x:x[1],reverse = True)[1:6]
    for i in similar_items:
        print(pivot_ratings.index[i[0]])
    
recommend('2001: A Space Odyssey (1968)')