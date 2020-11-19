import os
import time
import numpy as np
import tensorflow as tf


class Run(object):
    """
    This Run class handles initialising the trainer, loading data, creating the Tensorflow dataset and model training
    for the unsupervised representation model.

    Parameters
    ----------
    data_dir : str
        The directory containing the data. It should contain the following 3 memory-mapped Numpy arrays:
            entities_x.mmap: the entity types for the first entity in each sentence.
            entities_y.mmap: the entity types for the second entity in each sentence.
            contexts.mmap: the contexts in which each entity pair occurs.
    vocab : list[str]
        The list of valid tokens.
    entity_types : list[str]
        The list of valid entity types (effectively the vocabulary for the entities).
    num_data : int
        The number of data points in the above memory-mapped Numpy arrays.
    max_len : int
        The maximum length of the contexts.
    trainer : trainers.unsup.Trainer
        The trainer responsible for training the model.
    trainer_kwargs : dict
        Any keyword arguments with which to initialise the trainer.
    out_dir : str
        The directory where the outputs will be stored.
    pre_trained : bool
        Whether or not the model has already been trained.
    pre_trained_dir : str or None
        The directory from which to load pre-trained parameter values.
    """

    def __init__(self, data_dir, vocab, entity_types, num_data, max_len, trainer, trainer_kwargs, out_dir,
                 pre_trained=False, pre_trained_dir=None):

        self.vocab = vocab
        self.entity_types = entity_types

        self.num_data = num_data
        self.max_len = max_len

        self.strategy = tf.distribute.MirroredStrategy()

        with self.strategy.scope():
            self.trainer = trainer(strategy=self.strategy, **trainer_kwargs)

        self.entities_x, self.entities_y, self.contexts = self.load_data(data_dir)

        self.out_dir = out_dir

        self.pre_trained = pre_trained
        self.pre_trained_dir = pre_trained_dir

        trainable_variables = {v.name: v for v in self.trainer.trainable_variables}
        self.checkpoint = tf.train.Checkpoint(optimiser=self.trainer.optimiser, **trainable_variables)

        if self.pre_trained:
            self.checkpoint.restore(os.path.join(self.pre_trained_dir, 'saved_model'))

    def load_data(self, data_dir):
        """
        Load the required data into Numpy arrays.

        Parameters
        ----------
        data_dir : str
            The directory containing the data. It should contain the following 3 memory-mapped Numpy arrays:
                entities_x.mmap: the entity types for the first entity in each sentence.
                entities_y.mmap: the entity types for the second entity in each sentence.
                contexts.mmap: the contexts in which each entity pair occurs

        Returns
        -------
        entities_x : np.memmap
            The indices to `self.entity_types` for the first entity in each sentence.
        entities_y : np.memmap
            The indices to `self.entity_types` for the second entity in each sentence.
        contexts : np.memmap
            The indices to `self.vocab` for the contexts in which each entity pair occurs.
        """

        entities_x = np.memmap(os.path.join(data_dir, 'entities_x.mmap'), dtype=np.uint16, mode='r',
                               shape=(self.num_data,))
        entities_y = np.memmap(os.path.join(data_dir, 'entities_y.mmap'), dtype=np.uint16, mode='r',
                               shape=(self.num_data,))
        contexts = np.memmap(os.path.join(data_dir, 'contexts.mmap'), dtype=np.uint16, mode='r',
                             shape=(self.num_data, self.max_len))

        return entities_x, entities_y, contexts

    def create_dataset(self, n_batch, n_iter_warm_up):
        """
        Create the Tensorflow dataset to feed to the trainer.

        Parameters
        ----------
        n_batch : int
            The batch size.
        n_iter_warm_up : int
            The number of iterations over which the KL divergence will be annealed from 0 to 1.

        Returns
        -------
        tf.data.Dataset
        """

        def gen():

            i = 1

            while True:

                inds = np.random.choice(self.num_data, size=(n_batch,))

                x = np.int32(self.entities_x[inds])
                y = np.int32(self.entities_y[inds])
                c = np.int32(self.contexts[inds])

                if n_iter_warm_up is not None:
                    beta = np.float32(np.minimum(1., i / n_iter_warm_up))
                else:
                    beta = np.float32(1.)

                i += 1

                yield x, y, c, np.tile(beta, (n_batch,))

        return tf.data.Dataset.from_generator(gen, output_types=(tf.int32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=([None], [None], [None, self.max_len], [None])
                                              )

    def train(self, n_iter, n_batch, n_samples, n_iter_warm_up, save_freq=100000):
        """
        Train the model, saving the parameters at a fixed frequency using Tensorflow checkpoints.

        Parameters
        ----------
        n_iter : int
            The number of iterations for which to train the model.
        n_batch : int
            The training batch size.
        n_samples : int
            The number of latent samples to draw during training.
        n_iter_warm_up : None or int
            The number of iterations over which to anneal the KL divergence term of the ELBO.
        save_freq : int
            The frequency (in iterations) at which to save the parameter values.

        Returns
        -------
        elbo_kl : array
            The array of (elbo,kl) values for all iterations (for testing)
        """

        dataset_train = self.create_dataset(n_batch, n_iter_warm_up)
        dist_dataset_train = self.strategy.experimental_distribute_dataset(dataset_train)

        i = 1

        elbo_kl = np.zeros((n_iter, 2))

        with self.strategy.scope():
            for inputs in dist_dataset_train:
                start = time.perf_counter()

                elbo, kl = self.trainer.optimise(inputs, n_batch, n_samples)

                elbo_kl[i-1, 0] = elbo.numpy()
                elbo_kl[i-1, 1] = kl.numpy()

                print(
                    'Iteration ' + str(i) + ': objective = ' + str(elbo.numpy()) + ' kl = ' + str(kl.numpy()) +
                    ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)')

                if i % save_freq == 0:
                    self.checkpoint.write(os.path.join(self.out_dir, 'saved_model'))

                if i >= n_iter:
                    break

                i += 1

        self.checkpoint.write(os.path.join(self.out_dir, 'saved_model'))

        return elbo_kl
