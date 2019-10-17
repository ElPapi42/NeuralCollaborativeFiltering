import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
import zipfile
import requests

def EDA():
    #Downloads and extract Dataset to local, wait for download, i dont want to put a progress bar here sorry
    #You can run this on google colab for get faster downloads speeds
    if(not os.path.exists("./Datasets/MoviLens20M.zip")):

        resp = requests.get("http://files.grouplens.org/datasets/movielens/ml-20m.zip")

        os.mkdir("./Datasets")

        with open("./Datasets/MoviLens20M.zip", "wb") as f:
            f.write(resp.content)

        with zipfile.ZipFile("./Datasets/MoviLens20M.zip", "r") as zip_ref:
            zip_ref.extractall("./Datasets")

    #Loads Dataset, we only need ratings.csv and movies.csv files, we can drop timestamp and genres for now
    ratings_df = pd.read_csv("./Datasets/ml-20m/ratings.csv").drop(["timestamp"], axis=1)
    movies_df = pd.read_csv("./Datasets/ml-20m/movies.csv").drop(["genres"], axis=1)

    ml_df = ratings_df.merge(movies_df, on="movieId")

    ml_df = ml_df.reindex(columns=["userId", "movieId", "title", "rating"])
    print(ml_df.head())

    #Check info about the Dataset
    print(ml_df.info())

    #Check for NaNs
    print(ml_df.isna().sum())

    #List unique values of each column
    n_users = ml_df["userId"].max()
    n_movies = ml_df["movieId"].nunique()

    print("Unique Users: " + str(n_users))
    print("Unique Movies: " + str(n_movies))

    #Top movies with more rating count (dont confuse with more views or more rating score, but are correlated)
    count = ml_df["title"].value_counts()
    print(count[:15])

    #Normalize ratings
    ml_df["rating_norm"] = ml_df["rating"] / 5.0

    #Set Ids as categorical data
    ml_df["userId"] = ml_df["userId"].astype("category").cat.codes.values
    ml_df["movieId"] = ml_df["movieId"].astype("category").cat.codes.values
    print(ml_df.head())

    #Redimension Target data.
    users = ml_df["userId"].values
    movies = ml_df["movieId"].values
    ratings = ml_df["rating_norm"].values.reshape([-1, 1])

    #Create Datasets for train, evaluation and testing, and a full version of the dataset
    #Note: Value of shuffle buffer arguments is crucial for get well distributed dataset slices
    ml_ds = tf.data.Dataset.from_tensor_slices(({"userId":users, "movieId":movies}, ratings)).shuffle(200000)
    #full_ds = ml_ds
    eval_ds = ml_ds.take(10000).batch(10000)
    ml_ds = ml_ds.skip(10000)
    test_ds = ml_ds.take(10000).batch(10000)
    train_ds = ml_ds.skip(10000)

    #Check distributions of dataset slices, if both dist are close, we can inffer the distribution between the three datasets comes from the same distribution
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    plt.figure(figsize=(15, 3))

    plt.subplot(1, 3, 1)
    sns.distplot([sample["userId"] for sample, rating in test_ds], bins=100)
    sns.distplot([sample["userId"] for sample, rating in eval_ds], bins=100)

    plt.subplot(1, 3, 2)
    sns.distplot([sample["movieId"] for sample, rating in test_ds], bins=100)
    sns.distplot([sample["movieId"] for sample, rating in eval_ds], bins=100)

    plt.subplot(1, 3, 3)
    sns.distplot([rating for sample, rating in test_ds], bins=100)
    sns.distplot([rating for sample, rating in eval_ds], bins=100)

    plt.show()

    return train_ds, eval_ds, test_ds, n_users, n_movies