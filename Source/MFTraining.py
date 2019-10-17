import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from MatrixFactorizer import MatrixFactorizer
from EDA import EDA

train_ds, eval_ds, test_ds, n_users, n_movies = EDA()

#Matrix Factorizer Hyperparams
f_emb_dim = 16
f_lr = 0.0015
f_epochs = 10
f_batch_size = 40960

#Model instantiation
factorizer = MatrixFactorizer(n_users, n_movies, f_emb_dim)
factorizer.compile(tf.keras.optimizers.Adam(f_lr), tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
factorizer.train_on_batch(train_ds.batch(f_batch_size))
print(factorizer.summary())

#Model fitting
factorizer.fit(train_ds.batch(f_batch_size), epochs=f_epochs, validation_data=eval_ds)

#Test Performance of Factorizer
factorizer.evaluate(test_ds)

#Lets predict someones
pred = factorizer.predict(test_ds)
target = [target for sample, target in test_ds]

sns.distplot(pred, bins=10)
sns.distplot(target, bins=10)
plt.show()

#Above, we can see that the predictions distribution follow more or less the true distribution, but have margin for improvements
#Lets check using KLDivergence
print("KL Divergence: ", tf.keras.losses.KLDivergence()(target, pred).numpy())