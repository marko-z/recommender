import numpy as np
import pandas as pd
import sklearn
import sklearn.manifold
import collections
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
pd.options.display.float_format = '{:.3f}'.format

# (Heavily) inspired by:
# https://colab.research.google.com/github/google/eng-edu/blob/master/ml/recommendation-systems/recommendation-systems.ipynb

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

movie_cols = [
    'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
] + genre_cols


def initialize_movies():
    rating_data = pd.read_csv('ml-100k/u.data', usecols=[0, 1, 2], sep='\t', names=[
                              'user_id', 'movie_id', 'rating'], encoding='latin-1')

    movie_data = pd.read_csv(
        'ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')

    # movie_data: title, date, imdb, genre etc...
    # rating_data: user_id, movie_id, rating etc...
    # user_list:

    # Shifting ids to start with 0, they start as int64 and end up as object
    rating_data["movie_id"] = rating_data["movie_id"].apply(lambda x: str(x-1))
    rating_data["user_id"] = rating_data["user_id"].apply(lambda x: str(x-1))
    movie_data["movie_id"] = movie_data["movie_id"].apply(lambda x: str(x-1))

    # converting rating_data['rating'] from the initial int64 to float
    rating_data["rating"] = rating_data["rating"].apply(lambda x: float(x))

    # converting the 0 0 0 0 1 0 mess into a list of genres which the movie belongs
    # to

    # (1, 0, 0, 1, 0, 1),
    # (0, 1, 0, 1, 0, 0),
    # ...
    genre_ticks_for_each_movie = zip(
        *[movie_data[genre] for genre in genre_cols])

    relevant_genres = []
    # zip((1,0,0,1,0,1), (crime, thriller, fantasy, romance, show, etc.))
    # --> (crime, romance, etc.)
    for genre_ticks_of_particular_movie in genre_ticks_for_each_movie:
        relevant_genres.append(', '.join([genre for genre, tick in zip(
            genre_ticks_of_particular_movie, genre_cols) if tick == 1]))

    movie_data['combined_genres'] = relevant_genres

    user_count = rating_data['user_id'].nunique()
    # so this one has been giving the erronous 1664 instead of the correct 1682
    # movie_count = movie_data['title'].nunique()
    movie_count = movie_data.shape[0]

    print('rating_data BEFORE BEING SENT BY MOVIE DATAFRAME:')
    print(rating_data.head(10))
    print(f'user_count: {user_count}, movie_count: {movie_count}, movie_data.shape[0]: {movie_data.shape[0]}')
    print('################################################################')

    return rating_data, movie_data, user_count, movie_count

# so how can we insert a new rating and preserve the new version of rating data
# in the app context?
def insert_new_rating(rating_data, user_id, item_id, rating):
    new_rating = (user_id, item_id, rating)
    # shouldn't it be .iloc? are we sure this appends at the end?
    rating_data.loc[len(rating_data)] = new_rating
    return rating_data

def build_model(rating_data, user_count, movie_count):

    # helper, not sure if needed cause the data is already split into train and test?
    def split_dataframe(df, holdout_fraction=0.1):
        test = df.sample(frac=holdout_fraction, replace=False)
        train = df[~df.index.isin(test.index)]
        return train, test

    # helper
    def build_rating_sparse_tensor(rating_data, user_count, movie_count):
        """
        Args:
            rating_data: a pd.DataFrame with `user_id`, `movie_id` and `rating` columns.
        Returns:
            a tf.SparseTensor representing the ratings matrix.
        """
        # wouldn't this be really inconvenient if I start populating the
        # rating_data with large id numbers? Should I stick with low numbers
        # (i.e choose the next one in line?)
        indices = rating_data[['user_id', 'movie_id']].values
        values = rating_data['rating'].values
        return tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[user_count, movie_count])

    # helper
    def sparse_mean_square_error(sparse_ratings, user_embeddings, movie_embeddings):
        """
        Args:
        sparse_ratings: A SparseTensor rating matrix, of dense_shape [N, M]
        user_embeddings: A dense Tensor U of shape [N, k] where k is the embedding
            dimension, such that U_i is the embedding of user i.
        movie_embeddings: A dense Tensor V of shape [M, k] where k is the embedding
            dimension, such that V_j is the embedding of movie j.
        Returns:
        A scalar Tensor representing the MSE between the true ratings and the
            model's predictions.
        """
        predictions = tf.gather_nd(
            tf.matmul(user_embeddings, movie_embeddings, transpose_b=True),
            sparse_ratings.indices)
        loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
        return loss

    # helper

    class CFModel(object):
        """Simple class that represents a collaborative filtering model"""

        def __init__(self, embedding_vars, loss, metrics=None):
            """Initializes a CFModel.
            Args:
            embedding_vars: A dictionary of tf.Variables.
            loss: A float Tensor. The loss to optimize.
            metrics: optional list of dictionaries of Tensors. The metrics in each
                dictionary will be plotted in a separate figure during training.
            """
            self._embedding_vars = embedding_vars
            self._loss = loss
            self._metrics = metrics
            self._embeddings = {k: None for k in embedding_vars}
            self._session = None

        @property
        def embeddings(self):
            """The embeddings dictionary."""
            return self._embeddings

        def train(self, num_iterations=100, learning_rate=1.0,
                  optimizer=tf.train.GradientDescentOptimizer):
            """Trains the model.
            Args:
            iterations: number of iterations to run.
            learning_rate: optimizer learning rate.
            plot_results: whether to plot the results at the end of training.
            optimizer: the optimizer to use. Default to GradientDescentOptimizer.
            Returns:
            The metrics dictionary evaluated at the last iteration.
            """
            with self._loss.graph.as_default():
                opt = optimizer(learning_rate)
                train_op = opt.minimize(self._loss)
                local_init_op = tf.group(
                    tf.variables_initializer(opt.variables()),
                    tf.local_variables_initializer())
                if self._session is None:
                    self._session = tf.Session()
                    with self._session.as_default():
                        self._session.run(tf.global_variables_initializer())
                        self._session.run(tf.tables_initializer())
                        tf.train.start_queue_runners()

            with self._session.as_default():
                local_init_op.run()
                iterations = []
                metrics = self._metrics or ({},)
                metrics_vals = [collections.defaultdict(
                    list) for _ in self._metrics]

                # Train and append results.
                for i in range(num_iterations + 1):
                    _, results = self._session.run((train_op, metrics))
                    if (i % 10 == 0) or i == num_iterations:
                        print("\r iteration %d: " % i + ", ".join(
                            ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                            end='')
                        iterations.append(i)
                        for metric_val, result in zip(metrics_vals, results):
                            for k, v in result.items():
                                metric_val[k].append(v)

                for k, v in self._embedding_vars.items():
                    self._embeddings[k] = v.eval()

                # if plot_results:
                #     # Plot the metrics.
                #     num_subplots = len(metrics)+1
                #     fig = plt.figure()
                #     fig.set_size_inches(num_subplots*10, 8)
                #     for i, metric_vals in enumerate(metrics_vals):
                #         ax = fig.add_subplot(1, num_subplots, i+1)
                #         for k, v in metric_vals.items():
                #             ax.plot(iterations, v, label=k)
                #         ax.set_xlim([1, num_iterations])
                #         ax.legend()
                return results

    # helper
    def gravity(U, V):
        """Creates a gravity loss given two embedding matrices."""
        return 1. / (U.shape[0].value*V.shape[0].value) * tf.reduce_sum(
            tf.matmul(U, U, transpose_a=True) * tf.matmul(V, V, transpose_a=True))

    def build_regularized_model(
            rating_data, embedding_dim=3, regularization_coeff=.1, gravity_coeff=1.,
            init_stddev=0.1):
        """
        Args:
        ratings: the DataFrame of movie ratings.
        embedding_dim: The dimension of the embedding space.
        regularization_coeff: The regularization coefficient lambda.
        gravity_coeff: The gravity regularization coefficient lambda_g.
        Returns:
        A CFModel object that uses a regularized loss.
        """
        # Split the ratings DataFrame into train and test.
        train_ratings, test_ratings = split_dataframe(rating_data)
        # SparseTensor representation of the train and test datasets.
        A_train = build_rating_sparse_tensor(
            train_ratings, user_count, movie_count)
        A_test = build_rating_sparse_tensor(
            test_ratings, user_count, movie_count)
        U = tf.Variable(tf.random_normal(
            [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
        V = tf.Variable(tf.random_normal(
            [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))

        error_train = sparse_mean_square_error(A_train, U, V)
        error_test = sparse_mean_square_error(A_test, U, V)
        gravity_loss = gravity_coeff * gravity(U, V)
        regularization_loss = regularization_coeff * (
            tf.reduce_sum(U*U)/U.shape[0].value + tf.reduce_sum(V*V)/V.shape[0].value)
        total_loss = error_train + regularization_loss + gravity_loss
        losses = {
            'train_error_observed': error_train,
            'test_error_observed': error_test,
        }
        loss_components = {
            'observed_loss': error_train,
            'regularization_loss': regularization_loss,
            'gravity_loss': gravity_loss,
        }
        embeddings = {"user_id": U, "movie_id": V}

        return CFModel(embeddings, total_loss, [losses, loss_components])

    def train_model():
        reg_model = build_regularized_model(
            rating_data, regularization_coeff=0.1, gravity_coeff=1.0, embedding_dim=35,
            init_stddev=.05)
        reg_model.train(num_iterations=1000, learning_rate=20.)
        return reg_model

    return train_model()


def compute_scores(query_embedding, item_embeddings):
    u = query_embedding
    V = item_embeddings
    scores = u.dot(V.T)
    return scores


def user_recommendations(model, movie_data, rating_data, user_id, k=6):
    scores = compute_scores(
        model.embeddings["user_id"][user_id], model.embeddings["movie_id"])
    score_key = 'similarity'
    df = pd.DataFrame({
        score_key: list(scores),
        'item_id': movie_data['movie_id'],
        'titles': movie_data['title'],
        'genres': movie_data['all_genres'],
    })
    rated_movies = rating_data[rating_data['user_id']
                               == str(user_id)]["movie_id"].values
    df = df[df.item_id.apply(lambda movie_id: movie_id not in rated_movies)]
    return df.sort_values([score_key], ascending=False).head(k)


def movie_neighbors(model, movie_data, movie_id, k=6):
    scores = compute_scores(
        model.embeddings["movie_id"][movie_id], model.embeddings["movie_id"])
    sorted_items = movie_data
    sorted_items['score'] = pd.Series(scores, index=sorted_items.index)
    sorted_items = sorted_items.sort_values(['score'], ascending=False).head(k)
    return sorted_items


def find_movies(df, query):
    return df[df['title'].str.contains(query)]
