import tensorflow as tf

class DenseBlock(tf.keras.layers.Layer):
  "Dense + Dropout + BatchNorm"

  def __init__(self, units, dropout=0.1, l2=0.1):
    super(DenseBlock, self).__init__()

    self.bn = tf.keras.layers.BatchNormalization()
    self.drop = tf.keras.layers.Dropout(dropout)
    self.dense = tf.keras.layers.Dense(units, 
                                       "relu",
                                       kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2),
                                       kernel_constraint=tf.keras.constraints.UnitNorm())
    
  def call(self, inputs):
    X = self.bn(inputs)
    X = self.drop(X)
    X = self.dense(X)
    return X

#Model Definition
class Recommender(tf.keras.Model):
  """Scores the match between an user and a movie, higher scores mean more affinity o the user for the movie"""

  def __init__(self, users, movies, emb_dim, dense_struct, dropout=0.1, l2=0.001):
    super(Recommender, self).__init__()

    self.user_emb = tf.keras.layers.Embedding(users, emb_dim)
    self.user_flat = tf.keras.layers.Flatten()
    self.user_dense = DenseBlock(emb_dim, dropout, l2)

    self.movie_emb = tf.keras.layers.Embedding(movies, emb_dim)
    self.movie_flat = tf.keras.layers.Flatten()
    self.movie_dense = DenseBlock(emb_dim, dropout, l2)

    self.concat = tf.keras.layers.Concatenate()
    
    self.dense_list = list()
    for layer in dense_struct:
      self.dense_list.append(DenseBlock(layer, dropout, l2))

    self.dense_out = tf.keras.layers.Dense(units=1, 
                                           activation="sigmoid",
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2),
                                           kernel_constraint=tf.keras.constraints.UnitNorm())
    
  def call(self, inputs):

    X_user = self.user_emb(inputs["userId"])
    X_user = self.user_flat(X_user)
    X_user = self.user_dense(X_user)

    X_movie = self.movie_emb(inputs["movieId"])
    X_movie = self.movie_flat(X_movie)
    X_movie = self.movie_dense(X_movie)

    X = self.concat([X_user, X_movie])

    for layer in self.dense_list:
      X = layer(X)

    X = self.dense_out(X)
    return X