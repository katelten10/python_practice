import os

import pandas as pd
import numpy as np

DATA_PATH = "C:/Users/katel/Documents/Jupyter_practice/ratings.csv"
CACHE_DIR = "C:/Users/katel/Documents/Jupyter_practice/cache/"


#  Ratings data comes from: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip


def load_data(data_path):
    '''
    Load data
    :param data_path: data set path
    :param cache_path: data set cache path
    :return: users-items rating matrix
    '''
    # data set cache path
    cache_path = os.path.join(CACHE_DIR, "ratings_matrix.cache")

    print("Start loading data set...")
    if os.path.exists(cache_path):  # Determine whether a cache file exists or not
        print("Loading cache...")
        ratings_matrix = pd.read_pickle(
            cache_path)  # pickle makes it easier to reload the data set and don't need to recalculate or do the data format conversion
        print("Loading the dataset from the cache is complete")
    else:
        print("Loading new data...")

        # Set the type of data
        dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
        # We only choose the first three colunms: userId,movieId and rating
        ratings = pd.read_csv('ratings.csv', dtype=dtype, usecols=range(3))
        # PivotTable, which converts Movie IDs to column names, into a User-Movie scoring matrix
        ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
        ratings_matrix
        # Save to cache file
        ratings_matrix.to_pickle(cache_path)
        print("The dataset is loaded")
    return ratings_matrix


def compute_pearson_similarity(ratings_matrix, based="user"):
    '''
    Pearson correlation coefficient ( it's also a cosine similarity, but it centers the vectors first,
    and then calculates the cosine similarity after subtracting the mean of each vector a and b)

    :param ratings_matrix: users-items rating matrix
    :param based: "user" or "item"
    :return: similarity matrix
    '''
    user_similarity_cache_path = os.path.join(CACHE_DIR, "user_similarity.cache")
    item_similarity_cache_path = os.path.join(CACHE_DIR, "item_similarity.cache")

    # Similarity was calculated based on Pearson correlation coefficient
    # users similarity
    if based == "user":
        if os.path.exists(user_similarity_cache_path):
            print("The user similarity matrix is being loaded from the cache")
            similarity = pd.read_pickle(user_similarity_cache_path)
        else:
            print("Start calculating the user similarity matrix")
            similarity = ratings_matrix.T.corr()
            similarity.to_pickle(user_similarity_cache_path)

    # items similarity
    elif based == "item":
        if os.path.exists(item_similarity_cache_path):
            print("The item similarity matrix is loading from the cache")
            similarity = pd.read_pickle(item_similarity_cache_path)
        else:
            print("Start calculating the item similarity matrix")
            similarity = ratings_matrix.corr()
            similarity.to_pickle(item_similarity_cache_path)
    else:
        raise Exception("Unhandled 'based' Value: %s" % based)
    print("The similarity matrix has been calculated")
    return similarity


# Predict the rating of users
def predict(uid, iid, ratings_matrix, user_similar):
    '''
    Predicts the rating of a given user for a given item
    :param uid: userID
    :param iid: itemID
    :param ratings_matrix: user-item rating matrix
    :param user_similar: user-item similarity matrix
    :return: predicted rating value
    '''
    print("Begin to predict user <%d>'s rating for movie <%d>..." % (uid, iid))
    # 1. Find similar users of the uid user
    similar_users = user_similar[uid].drop([uid]).dropna()
    # Similar user filtering rule: Positive correlation users
    similar_users = similar_users.where(similar_users > 0).dropna()
    if similar_users.empty is True:
        raise Exception("User <%d> has no similar user" % uid)

    # 2. The nearest neighbor users with rating records for IID items were screened out from the nearest neighbor similar users of UID users
    ids = set(ratings_matrix[iid].dropna().index) & set(similar_users.index)
    finally_similar_users = similar_users.loc[list(ids)]
    # 3. combine the uid and his neiboring similarity userid's rating for certain item
    sum_up = 0  # numerator of the rating prediction formula
    sum_down = 0  # denominator of the rating prediction formula
    for sim_uid, similarity in finally_similar_users.iteritems():
        # Rating data for neighboring users
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # nerighboring users' ratings for iid items
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        # Calculate the numerator
        sum_up += similarity * sim_user_rating_for_item
        # Calculate the denominator
        sum_down += similarity

    # calculate the predicted rating value and return
    predict_rating = sum_up / sum_down
    print("predict user <%d>'s rating for movie <%d>：%0.2f" % (uid, iid, predict_rating))
    return round(predict_rating, 2)


def _predict_all(uid, item_ids, ratings_matrix, user_similar):
    '''
    predict all rating values
    :param uid: userid
    :param item_ids: list of itemid for prediction
    :param ratings_matrix: user-item rating matrix
    :param user_similar: user-user similarity
    :return: generator，return the prediction value one by one
    '''
    # one by one prediction
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating


def predict_all(uid, ratings_matrix, user_similar, filter_rule=None):
    '''
    predict all rating values，and prefilter according to the conditions
    :param filter_rule: Filter rule, only choose one of four, otherwise show exception："unhot","rated",["unhot","rated"],None
    :return: generator，return the prediction value one by one
    '''

    if not filter_rule:
        item_ids = ratings_matrix.columns
    elif isinstance(filter_rule, str) and filter_rule == "unhot":
        '''filter unhot movies'''
        # Calculate the ratings for each movie
        count = ratings_matrix.count()
        # Films with more than 10 ratings are filtered as popular movies
        item_ids = count.where(count > 10).dropna().index
    elif isinstance(filter_rule, str) and filter_rule == "rated":
        '''Filters for movies rated by users'''
        # Get a record of the user's ratings for all movies
        user_ratings = ratings_matrix.loc[uid]
        # The ratings range from 1 to 5. Anything less than 6 is rated. others are unrated
        _ = user_ratings < 6
        item_ids = _.where(_ == False).dropna().index
    elif isinstance(filter_rule, list) and set(filter_rule) == set(["unhot", "rated"]):
        '''filter unhot movies and movies already rated by users'''
        count = ratings_matrix.count()
        ids1 = count.where(count > 10).dropna().index

        user_ratings = ratings_matrix.loc[uid]
        _ = user_ratings < 6
        ids2 = _.where(_ == False).dropna().index
        # Take the intersection of the two
        item_ids = set(ids1) & set(ids2)
    else:
        raise Exception("Invalid filter parameter")

    yield from _predict_all(uid, item_ids, ratings_matrix, user_similar)


def top_k_rs_result(k):
    ratings_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(ratings_matrix, based="user")
    results = predict_all(1, ratings_matrix, user_similar, filter_rule=["unhot", "rated"])
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]


if __name__ == '__main__':
    ratings_matrix = load_data(DATA_PATH)
    user_similar = compute_pearson_similarity(ratings_matrix, based="user")
    print(user_similar)
    item_similar = compute_pearson_similarity(ratings_matrix, based="item")
    print(item_similar)

    for result in predict_all(1, ratings_matrix, user_similar, filter_rule=["unhot", "rated"]):
        print(result)

    from pprint import pprint
    result = top_k_rs_result(20)
    pprint(result)

