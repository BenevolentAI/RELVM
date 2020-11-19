import os
import tensorflow as tf


class MaximumLikelihoodMentionLevel(object):
    """
    This trainer is for training classification models using the maximum likelihood objective, classifying the
    relationship type between a pair of entities and the context in which they appear. This trainer assumes that there
    is a pre-trained unsupervised latent variable model from which to draw samples of a representation of the entity
    pair and context.

    Parameters
    ----------
    optimiser : tf.keras.optimizers.Optimizer
        The optimiser used to update the model parameters.
    optimiser_kwargs : dict
        Any keyword arguments with which to initialise the optimiser.
    trainer_unsup : trainers.unsup.Trainer
        The trainer used for the unsupervised representation model.
    model : models.sup.classification.Model
        The classification model to be trained.
    model_kwargs : dict
        Any keyword arguments with which to initialise the model.
    """

    def __init__(self, optimiser, optimiser_kwargs, trainer_unsup, trainer_unsup_kwargs, trainer_unsup_pre_trained_dir,
                 model, model_kwargs, fine_tune_unsup=False):

        self.optimiser = optimiser(**optimiser_kwargs)

        self.trainer_unsup = trainer_unsup(**trainer_unsup_kwargs)
        trainer_unsup_trainable_variables = {v.name: v for v in self.trainer_unsup.trainable_variables}
        checkpoint_unsup = tf.train.Checkpoint(optimiser=self.trainer_unsup.optimiser,
                                               **trainer_unsup_trainable_variables)
        checkpoint_unsup.restore(os.path.join(trainer_unsup_pre_trained_dir, 'saved_model'))

        self.model = model(**model_kwargs)

        self.trainable_variables = self.model.trainable_variables

        if fine_tune_unsup:
            self.trainable_variables += self.trainer_unsup.trainable_variables

    def log_p_r(self, x, y, c, r, n_samples, training=False):
        """
        Compute the log likelihood of a set of labels `r` given the entity pairs `x` and `y` and the contexts `c`, i.e.
        log(p(r|x,y,c)).

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        c : tf.Tensor(tf.int32)
            Shape : (N, L)
        r : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples, per data point, to draw of the representation from the unsupervised model.
        training : bool, optional

        Returns
        -------
        log_p_r : tf.Tensor(tf.float32)
            Shape : (N)
        """

        z = self.trainer_unsup.get_samples_z(x, y, c, n_samples)

        log_p_r = self.model.log_p_r(z, r, n_samples, training=training)

        return log_p_r

    @tf.function
    def objective(self, x, y, c, r, n_samples):
        """
        Tensorflow function to compute the average log likelihood per element in the batch.

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        c : tf.Tensor(tf.int32)
            Shape : (N, L)
        r : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples of z to draw.

        Returns
        -------
        log_p_r : tf.Tensor(tf.float32)
            Shape : ()
        """

        log_p_r = self.log_p_r(x, y, c, r, n_samples, training=False)

        log_p_r = tf.reduce_mean(log_p_r)

        return log_p_r

    @tf.function
    def optimise(self, x, y, c, r, n_samples):
        """
        Tensorflow function to update the model parameters, with the objective being the average log likelihood per
        element in the batch.

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        c : tf.Tensor(tf.int32)
            Shape : (N, L)
        r : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples of z to draw.

        Returns
        -------
        log_p_r : tf.Tensor(tf.float32)
            Shape : ()
        """

        with tf.GradientTape() as tape:
            log_p_r = self.log_p_r(x, y, c, r, n_samples, training=True)

            log_p_r = tf.reduce_mean(log_p_r)

            loss = -log_p_r

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))

        return log_p_r

    @tf.function
    def classify(self, x, y, c, n_samples):
        """
        Tensorflow function to predict the class of each element in a batch.

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
        log_probs : tf.Tensor(tf.float32)
            Shape : (N, C)
        r_pred : tf.Tensor(tf.int32)
            Shape : (N)
        """

        z = self.trainer_unsup.get_samples_z(x, y, c, n_samples)

        log_probs, r_pred = self.model.predict(z, n_samples)

        return log_probs, r_pred


class MaximumLikelihoodPairLevel(object):
    """
    This trainer is for training classification models using the maximum likelihood objective, classifying the
    relationship type between a pair of entities independently of the context in which they appear. This trainer assumes
    that there is a pre-trained unsupervised latent variable model from which to draw samples of a representation of the
    entity pair and context.

    Parameters
    ----------
    optimiser : tf.keras.optimizers.Optimizer
        The optimiser used to update the model parameters.
    optimiser_kwargs : dict
        Any keyword arguments with which to initialise the optimiser.
    trainer_unsup : trainers.unsup.Trainer
        The trainer used for the unsupervised representation model.
    model : models.sup.classification.Model
        The classification model to be trained.
    model_kwargs : dict
        Any keyword arguments with which to initialise the model.
    """

    def __init__(self, optimiser, optimiser_kwargs, trainer_unsup, trainer_unsup_kwargs, trainer_unsup_pre_trained_dir,
                 model, model_kwargs, fine_tune_unsup=False):

        self.optimiser = optimiser(**optimiser_kwargs)

        self.trainer_unsup = trainer_unsup(**trainer_unsup_kwargs)
        trainer_unsup_trainable_variables = {v.name: v for v in self.trainer_unsup.trainable_variables}
        checkpoint_unsup = tf.train.Checkpoint(optimiser=self.trainer_unsup.optimiser,
                                               **trainer_unsup_trainable_variables)
        checkpoint_unsup.restore(os.path.join(trainer_unsup_pre_trained_dir, 'saved_model'))

        self.model = model(**model_kwargs)

        self.trainable_variables = self.model.trainable_variables

        if fine_tune_unsup:
            self.trainable_variables += self.trainer_unsup.trainable_variables

    def log_p_r(self, x, y, r, n_samples, training=False):
        """
        Compute the log likelihood of a set of labels `r` given the entity pairs `x` and `y`, i.e. log(p(r|x,y)).

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        r : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples, per data point, to draw of the representation from the unsupervised model.
        training : bool, optional

        Returns
        -------
        log_p_r : tf.Tensor(tf.float32)
            Shape : (N)
        """

        z = self.trainer_unsup.get_samples_z_prior(x, y, n_samples)

        log_p_r = self.model.log_p_r(z, r, n_samples, training=training)

        return log_p_r

    @tf.function
    def objective(self, x, y, r, n_samples):
        """
        Tensorflow function to compute the average log likelihood per element in the batch.

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        r : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples of z to draw.

        Returns
        -------
        log_p_r : tf.Tensor(tf.float32)
            Shape : ()
        """

        log_p_r = self.log_p_r(x, y, r, n_samples, training=False)

        log_p_r = tf.reduce_mean(log_p_r)

        return log_p_r

    @tf.function
    def optimise(self, x, y, r, n_samples):
        """
        Tensorflow function to update the model parameters, with the objective being the average log likelihood per
        element in the batch.

        Parameters
        ----------
        x : tf.Tensor(tf.int32)
            Shape : (N)
        y : tf.Tensor(tf.int32)
            Shape : (N)
        r : tf.Tensor(tf.int32)
            Shape : (N)
        n_samples : int
            The number of samples of z to draw.

        Returns
        -------
        log_p_r : tf.Tensor(tf.float32)
            Shape : ()
        """

        with tf.GradientTape() as tape:
            log_p_r = self.log_p_r(x, y, r, n_samples, training=True)

            log_p_r = tf.reduce_mean(log_p_r)

            loss = -log_p_r

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimiser.apply_gradients(zip(gradients, self.trainable_variables))

        return log_p_r

    @tf.function
    def classify(self, x, y, n_samples):
        """
        Tensorflow function to predict the class of each element in a batch.

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
        log_probs : tf.Tensor(tf.float32)
            Shape : (N, C)
        r_pred : tf.Tensor(tf.int32)
            Shape : (N)
        """

        z = self.trainer_unsup.get_samples_z_prior(x, y, n_samples)

        log_probs, r_pred = self.model.predict(z, n_samples)

        return log_probs, r_pred
