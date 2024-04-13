import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Load ratings, movies and tags data
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')

# splitting a single genre column into multiple columns
movies['genres'] = movies['genres'].apply( lambda x: x.split('|'))

movie_tags = tags.groupby('movieId')['tag'].agg(list).reset_index()
movies_final = movies.merge(movie_tags, on='movieId', how='left')
# Replace NaN with blank
movies_final['tag'] = movies_final['tag'].fillna('')

movies_final['genres'] = movies_final['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_final['tag'] = movies_final['tag'].apply(lambda x:[i.replace(" ","") for i in x])

# Extract the release year from the movie title
pattern = r'\((\d{4})\)$'
movies_final['release-year'] = movies_final['title'].str.extract(pattern)

# Convert release year to array format
movies_final['release-year'] = movies_final['release-year'].apply(lambda year: [year])

movies_final['final_tag'] = movies_final['genres'] + movies_final['tag'] + movies_final['release-year']
movies_df = movies_final[['movieId', 'title', 'final_tag']]

# Convert the 'final_tag' column from list format to string format
movies_df['final_tag'] = movies_df['final_tag'].apply(lambda x: ' '.join(map(str, x)))


ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
movies_df['final_tag'].apply(stem)
cv = CountVectorizer(max_features = 1600, stop_words = 'english')
vectors = cv.fit_transform(movies_df['final_tag']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie_name):
    movie_index = movies_df[movies_df['title'] == movie_name].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key = lambda x:x[1])[1:6]
    for i in movies_list:
        print(movies_df.iloc[i[0]].title)
    return movies_list

recommend('Toy Story (1995)')
