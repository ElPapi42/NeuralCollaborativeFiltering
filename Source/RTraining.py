import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Recommender import Recommender
from EDA import EDA

train_ds, eval_ds, test_ds, n_users, n_movies = EDA()

#NCF Hyperparams
r_emb_dim = 32
r_lr = 0.0005
r_epochs = 10
r_l2 = 0.0000
r_dropout = 0.0
r_batch_size = 40960
r_dense_struct = [16, 4]

#Model instantiation
recommender = Recommender(n_users, n_movies, r_emb_dim, r_dense_struct, r_dropout, r_l2)
recommender.compile(tf.keras.optimizers.Adam(r_lr), tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
recommender.train_on_batch(train_ds.batch(r_batch_size))
print(recommender.summary())

#Model fit
recommender.fit(train_ds.batch(r_batch_size), epochs=r_epochs, validation_data=eval_ds)

recommender.evaluate(test_ds)

#Lets predict someones
pred = recommender.predict(test_ds)
target = [target for sample, target in test_ds]

sns.distplot(pred, bins=10)
sns.distplot(target, bins=10)
plt.show()

#Lets check using KLDivergence
print("KL Divergence: ", tf.keras.losses.KLDivergence()(target, pred).numpy())