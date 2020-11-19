import tensorflow as tf


class Classification(object):
    """
    Model the probability of the relation type given a latent representation z, i.e. p(r|z).

    Parameters
    ----------
    z_dim : int
        The dimensionality of the latent variable, which is used as the input to the classification model.
    n_classes : int
        The number of possible classes (i.e. relation types).
    nn_kwargs : dict
        The hyperparameters of the network for computing the parameters of p(r|z).
    """

    def __init__(self, z_dim, n_classes, nn_kwargs):

        self.z_dim = z_dim
        self.n_classes = n_classes

        self.model = self.nn(**nn_kwargs)

        self.trainable_variables = self.model.trainable_variables

        print('Classification model trainable variables = ' +
              str(sum([tf.size(v).numpy() for v in self.trainable_variables])))

    def nn(self, **kwargs):
        """
        Construct the Keras neural network to compute the parameters of p(r|z).

        Parameters
        ----------
        **kwargs
            The hyperparameters of the network.

        Returns
        -------
        tf.keras.Model
        """

        ff_depth = kwargs['ff_depth']
        ff_units = kwargs['ff_units']
        ff_activation = kwargs['ff_activation']

        input_z = tf.keras.Input(shape=(self.z_dim,))

        h = input_z

        for d in range(ff_depth):
            h = tf.keras.layers.Dense(units=ff_units, activation=ff_activation, use_bias=False)(h)
            h = tf.keras.layers.Concatenate()([h, input_z])

        out = tf.keras.layers.Dense(units=self.n_classes, activation='softmax', use_bias=False)(h)

        return tf.keras.Model(inputs=input_z, outputs=out)

    def log_p_r(self, z, r, n_samples, training=False):
        """
        Compute log(p(r|z)).

        Parameters
        ----------
        z : tf.Tensor(tf.float32)
            Shape : (N, S, Z)
        r : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples of z.
        training : bool, optional

        Returns
        -------
        log_p_r : tf.Tensor(tf.float32)
            Shape : (N)
        """

        N = r.shape[0]

        z_flat = tf.reshape(z, (N * n_samples, self.z_dim))  # (N*S, Z)

        nn_out = self.model(z_flat, training=training)  # (N*S, R)
        nn_out = tf.reshape(nn_out, (N, n_samples, self.n_classes))  # (N, S, R)

        log_probs = tf.reduce_mean(tf.math.log(nn_out + 1.e-5), axis=1)  # (N, R)

        r_one_hot = tf.one_hot(r, self.n_classes)  # (N, R)

        log_p_r = tf.reduce_sum(r_one_hot * log_probs, axis=-1)  # (N)

        return log_p_r

    def predict(self, z, n_samples):
        """
        Predict the class for each element in a batch.

        Parameters
        ----------
        z : tf.Tensor(tf.float32)
            Shape : (N, S, Z)
        n_samples : int
            The number of samples of z.

        Returns
        -------
        r_pred : tf.Tensor(tf.int32)
            Shape : (N)
        """

        N = z.shape[0]

        z_flat = tf.reshape(z, (N * n_samples, self.z_dim))  # (N*S, Z)

        nn_out = self.model(z_flat, training=False)  # (N*S, R)
        nn_out = tf.reshape(nn_out, (N, n_samples, self.n_classes))  # (N, S, R)

        log_probs = tf.reduce_mean(tf.math.log(nn_out + 1.e-5), axis=1)  # (N, R)

        r_pred = tf.math.argmax(log_probs, axis=-1, output_type=tf.int32)  # (N)

        return log_probs, r_pred
