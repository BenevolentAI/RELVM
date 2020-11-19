import tensorflow as tf


class GaussianBiLSTM(object):
    """
    Approximate the posterior distribution of a latent variable given an entity pair and context, i.e. q(u|x,y,c).

    Parameters
    ----------
    max_len : int
        The maximum length of the contexts.
    emb_dim : int
        The dimensionality of the static token embeddings.
    u_dim : int
        The dimensionality of the latent variable.
    nn_kwargs : dict
        The hyperparameters of the network for computing the mean and variance of q(u|x,y,c).
    """

    def __init__(self, max_len, emb_dim, u_dim, nn_kwargs):

        self.max_len = max_len
        self.emb_dim = emb_dim
        self.u_dim = u_dim

        self.model = self.nn(**nn_kwargs)

        self.trainable_variables = self.model.trainable_variables

        print('Recognition model trainable variables = ' +
              str(sum([tf.size(v).numpy() for v in self.trainable_variables])))

    def nn(self, **kwargs):
        """
        Construct the Keras neural network to compute the the mean and variance of q(u|x,y,c).

        Parameters
        ----------
        **kwargs
            lstm_depth : int
                The number of (bidirectional) LSTM layers.
            lstm_units : int
                The number of units in the LSTM layers.
            ff_depth : int
                The depth of the feedforward part of the network.
            ff_units : int
                The number of units in the feedforward layers.
            ff_activation : str
                The nonlinearity for the feedforward part of the network.

        Returns
        -------
        tf.keras.Model
        """

        lstm_depth = kwargs['lstm_depth']
        lstm_units = kwargs['lstm_units']

        ff_depth = kwargs['ff_depth']
        ff_units = kwargs['ff_units']
        ff_activation = kwargs['ff_activation']

        input_x = tf.keras.Input(shape=(self.emb_dim,))
        input_y = tf.keras.Input(shape=(self.emb_dim,))
        input_c = tf.keras.Input(shape=(self.max_len, self.emb_dim))
        input_c_mask = tf.keras.Input(shape=(self.max_len,), dtype=tf.bool)

        h_c_fwd = input_c
        h_c_bwd = input_c

        for d in range(lstm_depth):
            h_c_fwd = tf.keras.layers.LSTM(units=lstm_units)(input_c, mask=input_c_mask)
            h_c_bwd = tf.keras.layers.LSTM(units=lstm_units, go_backwards=True)(input_c, mask=input_c_mask)

        h_c = tf.keras.layers.Concatenate(axis=-1)([h_c_fwd, h_c_bwd])

        x = tf.keras.layers.Lambda(lambda l: l / tf.norm(l, axis=-1, keepdims=True))(input_x)
        y = tf.keras.layers.Lambda(lambda l: l / tf.norm(l, axis=-1, keepdims=True))(input_y)

        xy = tf.keras.layers.Multiply()([x, y])

        h = tf.keras.layers.Concatenate()([x, y, xy, h_c])

        for d in range(ff_depth):
            h = tf.keras.layers.Dense(units=ff_units, activation=ff_activation)(h)

        mean = tf.keras.layers.Dense(units=self.u_dim)(h)
        var = tf.keras.layers.Dense(units=self.u_dim, activation=tf.exp)(h)

        return tf.keras.Model(inputs=[input_x, input_y, input_c, input_c_mask], outputs=[mean, var])

    def get_params_and_samples(self, x, y, c, c_mask, n_samples, training=False):
        """
        Compute the mean and variance of the distribution q(u|x,y,c) and draw samples from it.

        Parameters
        ----------
        x : tf.Tensor(tf.float32)
            Shape : (N, E)
        y : tf.Tensor(tf.float32)
            Shape : (N, E)
        c : tf.Tensor(tf.float32)
            Shape : (N, L, E)
        c_mask: tf.Tensor(tf.bool)
            Shape : (N, L)
        n_samples : int
            The number of samples to draw.
        training : bool, optional

        Returns
        -------
        params : list[tf.Tensor(tf.float32)]
            The mean and variance of the distribution q(u|x,y,c).
        samples : tf.Tensor(tf.float32)
            Samples from q(u|x,y,c).
        """

        N = x.shape[0]

        u_mean, u_var = self.model([x, y, c, c_mask], training=training)  # (N, U), (N, U)

        u_mean_rep = tf.tile(tf.reshape(u_mean, (N, 1, self.u_dim)), (1, n_samples, 1))  # (N, S, U)
        u_var_rep = tf.tile(tf.reshape(u_var, (N, 1, self.u_dim)), (1, n_samples, 1))  # (N, S, U)

        e = tf.random.normal(u_mean_rep.shape)  # (N, S, U)
        u = u_mean_rep + (tf.sqrt(u_var_rep) * e)  # (N, S, U)

        return [u_mean, u_var], u
