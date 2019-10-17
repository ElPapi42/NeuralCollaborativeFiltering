import tensorflow as tf

#Model Definition
class MatrixFactorizer(tf.keras.Model):
  """This model will be used for optimize the embeddings, later will be discarded, just keeping the embedding layers weights"""

  def __init__(self, users, movies, emb_dim):
    super(MatrixFactorizer, self).__init__()

    self.user_emb = tf.keras.layers.Embedding(users, emb_dim)
    self.user_flat = tf.keras.layers.Flatten()

    self.movie_emb = tf.keras.layers.Embedding(movies, emb_dim)
    self.movie_flat = tf.keras.layers.Flatten()

    self.dot = tf.keras.layers.Dot(axes=1)

  def call(self, inputs):
    X_user = self.user_emb(inputs["userId"])
    X_user = self.user_flat(X_user)

    X_movie = self.movie_emb(inputs["movieId"])
    X_movie = self.movie_flat(X_movie)

    X = self.dot([X_user, X_movie])
    return X