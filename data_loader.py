import pandas as pd
import numpy as np
# Make display smaller
def dataloader():
    pd.options.display.max_rows = 10
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_csv('ml-1m/users.dat', sep='::',
    header=None, names=unames, engine='python',encoding='ISO-8859-1')
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::',
    header=None, names=rnames, engine='python',encoding='ISO-8859-1')
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_csv('ml-1m/movies.dat', sep='::',
    header=None, names=mnames, engine='python',encoding='ISO-8859-1')
    return users,ratings,movies