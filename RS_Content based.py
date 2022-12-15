import re
from datetime import datetime
from data_loader import dataloader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pickle
from sklearn.metrics import mean_squared_error
import pandas as pd
import csv

users,ratings,movies = dataloader()
train_ratings, validation_ratings = train_test_split(
    ratings, test_size=0.1, random_state=42
)
users_in_validation = validation_ratings["user_id"].unique()
all_users = users["user_id"].unique()

print(f"There are {len(users_in_validation)} users in validation set.")
print(f"Total number of users: {len(all_users)}")

movie_index_by_id = {id: i for i, id in enumerate(movies["movie_id"])}
genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
genre_index_by_name = {name:i for i, name in enumerate(genres)}

import numpy as np
# build binary array for movie genres
movie_features = np.zeros((len(movies), len(genres)))
for i, movie_genres in enumerate(movies["genres"]):
    for genre in movie_genres.split("|"):        
        genre_index = genre_index_by_name[genre]
        movie_features[i, genre_index] = 1

def train_user_model(user_id):
    user_ratings = train_ratings[train_ratings["user_id"] == user_id]
    movie_indexes = [
        movie_index_by_id[movie_id] for movie_id in user_ratings["movie_id"]
    ]
    train_data = movie_features[movie_indexes]
    train_label = user_ratings["rating"]
    model = Ridge(alpha=0.1)
    model.fit(train_data, train_label)
    return model


# build model for each user
# user_model_dict = {}
# for user_id in users["user_id"].unique():
#     user_model_dict[user_id] = train_user_model(user_id)
filename = 'finalized_model.sav'
user_model_dict = pickle.load(open(filename, 'rb'))
def predict(user_id, movie_id):
    movie_feature = movie_features[movie_index_by_id[movie_id]].reshape((1, -1))
    pred = user_model_dict[user_id].predict(movie_feature)
    return min(max(pred, 1), 5)
def eval_rmse(ratings: pd.DataFrame) -> float:
    predictions = np.zeros(len(ratings))
    for index, row in enumerate(ratings.itertuples(index=False)):
        predictions[index] = predict(row[0], row[1])
    rmse = mean_squared_error(ratings["rating"], predictions, squared=False)
    return rmse
    
# print(f"RMSE train: {eval_rmse(train_ratings)}")
# print(f"RMSE validation: {eval_rmse(validation_ratings)}")
k = []
user_id = 160
for i in users["user_id"].unique():
    x = []
    for genre, coef in zip(genres, user_model_dict[i].coef_):
        x.append(coef)
    k.append(x)
print(k[160])
p = pd.DataFrame(columns=genres, data=k, index=users["user_id"] )
p.to_csv(r'u_dict.csv',index= True)

# print(predict(1, 1193))