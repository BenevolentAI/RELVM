import tensorflow as tf


class ConditionalInfiniteMixtureLSTMAutoregressive(object):
    """
    Model the probability of a context given the pair of entities which it contains, i.e.
    p(c|x,y) = ∫ p(z|x,y) p(c|z) dz with p(z|x,y) being an infinite Gaussian mixture model.

    Parameters
    ----------
    max_len : int
        The maximum length of the contexts.
    emb_dim : int
        The dimensionality of the static token embeddings.
    u_dim : int
        The dimensionality of the Gaussian responsibility vector.
    z_dim : int
        The dimensionality of the relationship vector.
    nn_z_kwargs : dict
        The hyperparameters of the network for computing the parameters of p(z|x,y).
    nn_context_kwargs : dict
        The hyperparameters of the network for computing the parameters of p(c|x,y,z).
    """

    def __init__(self, max_len, emb_dim, u_dim, z_dim, nn_z_kwargs, nn_context_kwargs):

        self.max_len = max_len
        self.emb_dim = emb_dim
        self.u_dim = u_dim
        self.z_dim = z_dim

        self.model_z = self.nn_z(**nn_z_kwargs)
        self.model_context = self.nn_context(**nn_context_kwargs)

        self.trainable_variables = self.model_z.trainable_variables + self.model_context.trainable_variables

        print('Generative model trainable variables = ' +
              str(sum([tf.size(v).numpy() for v in self.trainable_variables])))

    def nn_z(self, **kwargs):
        """
        Construct the Keras neural network to compute the the parameters of p(z|x,y,u).

        Parameters
        ----------
        **kwargs
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

        ff_depth = kwargs['ff_depth']
        ff_units = kwargs['ff_units']
        ff_activation = kwargs['ff_activation']

        input_x = tf.keras.Input(shape=(self.emb_dim,))
        input_y = tf.keras.Input(shape=(self.emb_dim,))
        input_u = tf.keras.Input(shape=(self.u_dim,))

        x = tf.keras.layers.Lambda(lambda l: l / tf.norm(l, axis=-1, keepdims=True))(input_x)
        y = tf.keras.layers.Lambda(lambda l: l / tf.norm(l, axis=-1, keepdims=True))(input_y)

        xy = tf.keras.layers.Multiply()([x, y])

        h = tf.keras.layers.Concatenate()([x, y, xy, input_u])

        for d in range(ff_depth):
            h = tf.keras.layers.Dense(units=ff_units, activation=ff_activation)(h)
            h = tf.keras.layers.Concatenate()([h, input_u])

        mean = tf.keras.layers.Dense(units=self.z_dim,
                                     kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.))(h)
        var = tf.keras.layers.Dense(units=self.z_dim, activation=tf.exp)(h)

        return tf.keras.Model(inputs=[input_x, input_y, input_u], outputs=[mean, var])

    def nn_context(self, **kwargs):
        """
        Construct the Keras neural network to compute the the parameters of p(c|z).

        Parameters
        ----------
        **kwargs
            ff_depth : int
                The depth of the feedforward part of the network.
            ff_units : int
                The number of units in the feedforward layers.
            ff_activation : str
                The nonlinearity for the feedforward part of the network.
            lstm_depth : int
                The depth of the LSTM.
            lstm_units : int
                The number of units in the LSTM layers.

        Returns
        -------
        tf.keras.Model
        """

        ff_depth = kwargs['ff_depth']
        ff_units = kwargs['ff_units']
        ff_activation = kwargs['ff_activation']

        lstm_depth = kwargs['lstm_depth']
        lstm_units = kwargs['lstm_units']

        token_drop = kwargs['token_drop']

        input_x = tf.keras.Input(shape=(self.emb_dim,))
        input_y = tf.keras.Input(shape=(self.emb_dim,))
        input_z = tf.keras.Input(shape=(self.z_dim,))
        input_c = tf.keras.Input(shape=(self.max_len, self.emb_dim))

        h = input_z

        for d in range(ff_depth):
            h = tf.keras.layers.Dense(units=ff_units, activation=ff_activation)(h)

        h = tf.keras.layers.RepeatVector(n=self.max_len)(h)

        c_slice = tf.keras.layers.Lambda(lambda l: l[:, :-1])(input_c)
        c_pre_pad = tf.keras.layers.ZeroPadding1D(padding=(1, 0))(c_slice)

        if token_drop > 0:
            c_pre_pad = tf.keras.layers.Dropout(rate=token_drop, noise_shape=[None, self.max_len, 1])(c_pre_pad)

        h = tf.keras.layers.Concatenate()([h, c_pre_pad])

        z_rep = tf.keras.layers.RepeatVector(n=self.max_len)(input_z)

        for d in range(lstm_depth):
            h = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True)(h)
            h = tf.keras.layers.Concatenate()([h, z_rep])

        out = tf.keras.layers.Dense(units=self.emb_dim)(h)

        return tf.keras.Model(inputs=[input_x, input_y, input_z, input_c], outputs=out)

    def kl_u(self, mean_rec, var_rec):
        """
        Compute the KL divergence from q(u|x,y,c) to p(u).

        Parameters
        ----------
        mean_rec : tf.Tensor(tf.float32)
            Shape : (N, U)
            The mean of the variational Gaussian q(u|x,y,c).
        var_rec : tf.Tensor(tf.float32)
            Shape : (N, U)
            The variance of the variational Gaussian q(u|x,y,c).

        Returns
        -------
        kl : tf.Tensor(tf.float32)
            Shape : (N)
        """

        kl = 0.5 * (tf.reduce_sum(var_rec + (mean_rec ** 2) - tf.ones_like(mean_rec) - tf.math.log(var_rec), axis=1))

        return kl

    def get_samples_u(self, n_batch, n_samples):
        """
        Draw samples of u.

        Parameters
        ----------
        n_batch : int
            The batch size.
        n_samples : int
            The number of samples of u to draw.

        Returns
        -------
        z : tf.Tensor(tf.float32)
            Shape : (N, S, U)
        """

        u = tf.random.normal((n_batch, n_samples, self.u_dim))  # (N, S, U)

        return u

    def get_samples_z(self, u, x, y, n_samples, training=False):
        """
        Draw samples of z.

        Parameters
        ----------
        u : tf.Tensor(tf.float32)
            Shape : (N, S, U)
        x : tf.Tensor(tf.float32)
            Shape : (N, E)
        y : tf.Tensor(tf.float32)
            Shape : (N, E)
        n_samples : int
            The number of samples of z to draw.
        training : bool, optional

        Returns
        -------
        z : tf.Tensor(tf.float32)
            Shape : (N, S, Z)
        """

        N = x.shape[0]

        u_flat = tf.reshape(u, (N * n_samples, self.u_dim))  # (N*S, U)

        x_rep_flat = tf.reshape(tf.tile(tf.reshape(x, (N, 1, self.emb_dim)), (1, n_samples, 1)),
                                (N * n_samples, self.emb_dim))  # (N*S, E)
        y_rep_flat = tf.reshape(tf.tile(tf.reshape(y, (N, 1, self.emb_dim)), (1, n_samples, 1)),
                                (N * n_samples, self.emb_dim))  # (N*S, E)

        z_mean_flat, z_var_flat = self.model_z([x_rep_flat, y_rep_flat, u_flat], training=training)
        # (N*S, Z), (N*S, Z)

        z_mean = tf.reshape(z_mean_flat, (N, n_samples, self.z_dim))  # (N, S, Z)
        z_var = tf.reshape(z_var_flat, (N, n_samples, self.z_dim))  # (N, S, Z)

        e = tf.random.normal(z_mean.shape)  # (N, S, Z)
        z = z_mean + (tf.sqrt(z_var) * e)  # (N, S, Z)

        return z

    def log_p_c(self, u, x, y, c, c_mask, emb_matrix_words, n_samples, training=False):
        """
        Compute log(p(c|x,y)) where c is the context for the entity pair (x,y).

        Parameters
        ----------
        u : tf.Tensor(tf.float32)
            Shape : (N, S, U)
        x : tf.Tensor(tf.float32)
            Shape : (N, E)
        y : tf.Tensor(tf.float32)
            Shape : (N, E)
        c : tf.Tensor(tf.float32)
            Shape : (N, L, E)
        c_mask: tf.Tensor(tf.bool)
            Shape : (N, L)
        emb_matrix_words : tf.Tensor(tf.float32)
            Shape : (V, E) where V is the size of the vocabulary.
        n_samples : int
            The number of samples of z to draw.
        training : bool, optional

        Returns
        -------
        log_p_c : tf.Tensor(tf.float32)
            Shape : (N)
        """

        N = x.shape[0]

        x_rep_flat = tf.reshape(tf.tile(tf.reshape(x, (N, 1, self.emb_dim)), (1, n_samples, 1)),
                                (N * n_samples, self.emb_dim))  # (N*S, E)
        y_rep_flat = tf.reshape(tf.tile(tf.reshape(y, (N, 1, self.emb_dim)), (1, n_samples, 1)),
                                (N * n_samples, self.emb_dim))  # (N*S, E)

        z = self.get_samples_z(u, x, y, n_samples, training=training)

        z_flat = tf.reshape(z, (N * n_samples, self.z_dim))

        c_rep = tf.tile(tf.reshape(c, (N, 1, self.max_len, self.emb_dim)), (1, n_samples, 1, 1))  # (N, S, L, E)
        c_rep_flat = tf.reshape(c_rep, (N * n_samples, self.max_len, self.emb_dim))  # (N*S, L, E)

        nn_out = self.model_context([x_rep_flat, y_rep_flat, z_flat, c_rep_flat], training=training)  # (N*S, L, E)
        nn_out = tf.reshape(nn_out, (N, n_samples, self.max_len, self.emb_dim))  # (N, S, L, E)

        c_mask_rep = tf.tile(tf.reshape(c_mask, (N, 1, self.max_len)), (1, n_samples, 1))  # (N, S, L)

        log_pot = tf.reduce_sum(c_rep * nn_out, axis=3)  # (N, S, L)
        log_pot = tf.where(c_mask_rep, log_pot, tf.zeros_like(log_pot))  # (N, S, L)

        nn_out_flat = tf.reshape(nn_out, (N * n_samples * self.max_len, self.emb_dim))  # (N*S*L, E)
        log_norm_flat = tf.reduce_logsumexp(tf.matmul(nn_out_flat, emb_matrix_words, transpose_b=True), axis=1)
        # (N*S*L)
        log_norm = tf.reshape(log_norm_flat, (N, n_samples, self.max_len))  # (N, S, L)
        log_norm = tf.where(c_mask_rep, log_norm, tf.zeros_like(log_norm))  # (N, S, L)

        c_mask_rep_float = tf.cast(c_mask_rep, tf.float32)  # (N, S, L)

        log_p_c = tf.reduce_mean(tf.reduce_sum(log_pot - log_norm, axis=2) / tf.reduce_sum(c_mask_rep_float, axis=2),
                                 axis=1)  # (N)

        return log_p_c
