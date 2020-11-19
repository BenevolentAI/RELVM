import tensorflow as tf


class SGVB(object):
    """
    This trainer is for training latent variable models using stochastic gradient variational Bayes, maximising the
    probability of a set of contexts given the pair of entities which they contain.

    Parameters
    ----------
    strategy : tf.distribute.Strategy
        The Tensorflow strategy for distributed training.
    optimiser : tf.keras.optimizers.Optimizer
        The optimiser used to update the model parameters.
    optimiser_kwargs : dict
        Any keyword arguments with which to initialise the optimiser.
    emb_matrix_entities : np.array
        The initial values for the entity embedding matrix.
    emb_matrix_words : np.array
        The initial values for the word embedding matrix.
    gen_model : models.unsup.generative.Model
        The generative model to be trained.
    gen_model_kwargs : dict
        Any keyword arguments with which to initialise the generative model.
    rec_model : models.unsup.recognition.Model
        The recognition model to be used to train the generative model.
    rec_model_kwargs : dict
        Any keyword arguments with which to initialise the recognition model.
    """

    def __init__(self, strategy, optimiser, optimiser_kwargs, emb_matrix_entities, emb_matrix_words, pad_ind, gen_model,
                 gen_model_kwargs, rec_model, rec_model_kwargs):

        self.strategy = strategy

        self.optimiser = optimiser(**optimiser_kwargs)

        self.emb_matrix_entities = tf.Variable(emb_matrix_entities, name='emb_matrix_entities')
        self.emb_matrix_words = tf.Variable(emb_matrix_words, name='emb_matrix_words')
        self.pad_ind = pad_ind

        self.vocab_size = self.emb_matrix_words.shape[0]
        self.emb_dim = self.emb_matrix_words.shape[1]

        self.gen_model = gen_model(**gen_model_kwargs)
        self.rec_model = rec_model(**rec_model_kwargs)

        self.trainable_variables = self.gen_model.trainable_variables + self.rec_model.trainable_variables + \
            [self.emb_matrix_entities, self.emb_matrix_words]

    def get_samples_z(self, x, y, c, n_samples):
        """
        Get samples of z, by first drawing samples of u from the recognition model and then drawing samples of z
        conditioned on the entity pair x and y, and the samples u.

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        c : tf.Tensor(tf.int32)
            Shape : (N, L)
        n_samples : int
            The number of samples of z to draw.

        Returns
        -------
        z : tf.Tensor(tf.float32)
            Shape : (N, S, Z)
        """

        x_emb = tf.gather(self.emb_matrix_entities, x)
        y_emb = tf.gather(self.emb_matrix_entities, y)
        c_emb = tf.gather(self.emb_matrix_words, c)
        c_mask = tf.not_equal(c, self.pad_ind)

        u_params, u = self.rec_model.get_params_and_samples(x_emb, y_emb, c_emb, c_mask, n_samples, training=False)

        z = self.gen_model.get_samples_z(u, x_emb, y_emb, n_samples, training=False)

        return z

    def get_samples_z_prior(self, x, y, n_samples):
        """
        Get samples of z, by first drawing samples of u from the prior and then drawing samples of z conditioned on the
        entity pair x and y, and the samples u.

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples of z to draw.

        Returns
        -------
        z : tf.Tensor(tf.float32)
            Shape : (N, S, Z)
        """

        x_emb = tf.gather(self.emb_matrix_entities, x)
        y_emb = tf.gather(self.emb_matrix_entities, y)

        u = self.gen_model.get_samples_u(x.shape[0], n_samples)

        z = self.gen_model.get_samples_z(u, x_emb, y_emb, n_samples, training=False)

        return z

    def elbo_and_kl(self, x, y, c, n_samples, beta=None, training=False):
        """
        Compute a lower bound on the log likelihood of a set of contexts `c` containing the pairs of entities `x` and
        `y`, i.e. log(p(c|x,y)), as well as the KL divergence between the approximate posterior and prior distributions
        of the latent variable.

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        c : tf.Tensor(tf.int32)
            Shape : (N, L)
        n_samples : int
            The number of samples of u to draw.
        beta : tf.Tensor(tf.float32)
            The annealing temperature with which to multiply the KL divergence term of the ELBO.
        training : bool, optional

        Returns
        -------
        elbo : tf.Tensor(tf.float32)
            Shape : (N)
        kl : tf.Tensor(tf.float32)
            Shape : (N)
        """

        x_emb = tf.gather(self.emb_matrix_entities, x)
        y_emb = tf.gather(self.emb_matrix_entities, y)
        c_emb = tf.gather(self.emb_matrix_words, c)
        c_mask = tf.not_equal(c, self.pad_ind)

        u_params, u = self.rec_model.get_params_and_samples(x_emb, y_emb, c_emb, c_mask, n_samples, training=training)

        u_mean_rec, u_var_rec = u_params

        c_mask_float = tf.cast(c_mask, tf.float32)
        kl = self.gen_model.kl_u(u_mean_rec, u_var_rec) / tf.reduce_sum(c_mask_float, axis=1)

        log_p_c = self.gen_model.log_p_c(u, x_emb, y_emb, c_emb, c_mask, self.emb_matrix_words, n_samples,
                                         training=training)

        if beta is None:
            elbo = log_p_c - kl
        else:
            elbo = log_p_c - (beta * kl)

        return elbo, kl

    @tf.function
    def optimise(self, dist_inputs, n_batch, n_samples):
        """
        Tensorflow function to update the model parameters, with the objective being the average log likelihood per
        element in the batch.

        Parameters
        ----------
        dist_inputs
            The elements of the distributed Tensorflow dataset. This should contain four arrays:
                x : tf.Tensor(tf.int32)
                    Shape : (N)
                y : tf.Tensor(tf.int32)
                    Shape : (N)
                c : tf.Tensor(tf.int32)
                    Shape : (N, L)
                beta : tf.Tensor(tf.float32)
                    Shape : (N)
        n_batch : int
            The number of elements in each batch.
        n_samples : int
            The number of latent samples to draw.

        Returns
        -------
        elbo : tf.Tensor(tf.float32)
            Shape : ()
        kl : tf.Tensor(tf.float32)
            Shape : ()
        """

        def step_fn(inputs):
            x, y, c, beta = inputs

            with tf.GradientTape() as tape:
                elbo, kl = self.elbo_and_kl(x, y, c, n_samples, beta, training=True)

                loss = -tf.reduce_sum(elbo) / tf.cast(n_batch, tf.float32)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))

            return elbo, kl

        per_example_elbos, per_example_kls = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))

        sum_elbo = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_elbos, axis=0)
        sum_kl = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_kls, axis=0)

        mean_elbo = sum_elbo / n_batch
        mean_kl = sum_kl / n_batch

        return mean_elbo, mean_kl
