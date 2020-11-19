import os
import time

import numpy as np
import tensorflow as tf


def precision_recall_f1(labels, predictions, label_types, neg_label):
    """
    Compute the precision, recall and F1 score for a set of predictions.

    Parameters
    ----------
    labels : np.array
        The true labels.
    predictions : np.array
        The predicted labels.
    label_types : list[str]
        The list of valid label types.
    neg_label : str
        The label indicating that two entities are not related.

    Returns
    -------
    precision : float
    recall : float
    f1 : float
    """

    tp = 0
    fp = 0
    fn = 0

    for i in range(len(labels)):
        if labels[i] == predictions[i] and labels[i] != label_types.index(neg_label):
            tp += 1
        elif predictions[i] != label_types.index(neg_label) and labels[i] != predictions[i]:
            fp += 1
        elif predictions[i] == label_types.index(neg_label) and labels[i] != predictions[i]:
            fn += 1

    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def optimise_threshold(labels, scores, label_types, neg_label, min_threshold, max_threshold, num):
    """
    Compute the probability threshold which maximises the F1 score. By threshold, we mean that the model will only
    predict a label other than `neg_label` if the predicted probability is higher than the threshold.

    Parameters
    ----------
    labels : np.array
        The true labels for each data point.
    scores : np.array
        The predicted probabilities for each data point for each label.
    label_types : list[str]
        The list of valid label types.
    neg_label : str
        The label indicating that two entities are not related.
    min_threshold : float
        The smallest threshold to consider.
    max_threshold : float
        The largest threshold to consider.
    num : int
        The number of thresholds to evaluate.

    Returns
    -------
    optimal_threshold : float
        The optimal threshold.
    """

    thresholds = np.linspace(min_threshold, max_threshold, num)

    argmaxes = np.argmax(scores, axis=-1)
    maxes = scores[np.arange(len(argmaxes)), argmaxes]

    f1s = []

    for threshold in thresholds:
        predictions = [argmaxes[i] if maxes[i] >= threshold else label_types.index(neg_label)
                       for i in range(len(argmaxes))]

        _, _, f1 = precision_recall_f1(labels, predictions, label_types, neg_label)

        f1s.append(f1)

    optimal_threshold = thresholds[np.argmax(f1s)]

    return optimal_threshold


class MaximumLikelihoodMentionLevel(object):
    """
    This Run class handles initialising the trainer, loading data, creating the Tensorflow dataset and model training
    for mention-level classification.

    Parameters
    ----------
    data_dir : str
        The directory containing the data. It should contain the following 12 memory-mapped Numpy arrays:
            entities_x_{train,valid,test}.mmap: the entity types for the first entity in each sentence.
            entities_y_{train,valid,test}.mmap: the entity types for the second entity in each sentence.
            contexts_{train,valid,test}.mmap: the contexts in which each entity pair occurs.
            labels_{train,valid,test}.mmap: the label classifying the relation type between the two entities in the
            context in which they occur.
    vocab : list[str]
        The list of valid tokens.
    entity_types : list[str]
        The list of valid entity types (effectively the vocabulary for the entities).
    label_types : list[str]
        The list of valid label types.
    neg_label : str
        The label indicating that two entities are not related.
    num_data_train : int
        The number of training data points.
    num_data_valid : int
        The number of validation data points.
    num_data_test : int
        The number of test data points.
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

    Returns
    -------
    obj : float
        The training objective
    f1 : float
        The prediction f1 score
    """

    def __init__(self, data_dir, vocab, entity_types, label_types, neg_label, num_data_train, num_data_valid,
                 num_data_test, max_len, trainer, trainer_kwargs, out_dir, pre_trained=False, pre_trained_dir=None):

        self.vocab = vocab
        self.entity_types = entity_types
        self.label_types = label_types
        self.neg_label = neg_label

        self.num_data_train = num_data_train
        self.num_data_valid = num_data_valid
        self.num_data_test = num_data_test
        self.max_len = max_len

        self.trainer = trainer(**trainer_kwargs)

        self.entities_x_train, self.entities_y_train, self.contexts_train, self.labels_train, self.entities_x_valid, \
            self.entities_y_valid, self.contexts_valid, self.labels_valid, self.entities_x_test, self.entities_y_test, \
            self.contexts_test, self.labels_test = self.load_data(data_dir)

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
            The directory containing the data. It should contain the following 12 memory-mapped Numpy arrays:
                entities_x_{train,valid,test}.mmap: the entity types for the first entity in each sentence.
                entities_y_{train,valid,test}.mmap: the entity types for the second entity in each sentence.
                contexts_{train,valid,test}.mmap: the contexts in which each entity pair occurs.
                labels_{train,valid,test}.mmap: the label classifying the relation type between the two entities in the
                context in which they occur.

        Returns
        -------
        entities_x_train : np.memmap
            The indices to `self.entity_types` for the first entity in each sentence in the training data.
        entities_y_train : np.memmap
            The indices to `self.entity_types` for the second entity in each sentence in the training data.
        contexts_train : np.memmap
            The indices to `self.vocab` for the contexts in which each entity pair occurs in the training data.
        labels_train : np.memmap
            The indices to `self.label_types` for the labels for each sentence in the training data.
        entities_x_valid : np.memmap
            The indices to `self.entity_types` for the first entity in each sentence in the validation data.
        entities_y_valid : np.memmap
            The indices to `self.entity_types` for the second entity in each sentence in the validation data.
        contexts_valid : np.memmap
            The indices to `self.vocab` for the contexts in which each entity pair occurs in the validation data.
        labels_valid : np.memmap
            The indices to `self.label_types` for the labels for each sentence in the validation data.
        entities_x_test : np.memmap
            The indices to `self.entity_types` for the first entity in each sentence in the test data.
        entities_y_test : np.memmap
            The indices to `self.entity_types` for the second entity in each sentence in the test data.
        contexts_test : np.memmap
            The indices to `self.vocab` for the contexts in which each entity pair occurs in the test data.
        labels_test : np.memmap
            The indices to `self.label_types` for the labels for each sentence in the test data.
        """

        entities_x_train = np.memmap(os.path.join(data_dir, 'entities_x_train.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_train,))
        entities_y_train = np.memmap(os.path.join(data_dir, 'entities_y_train.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_train,))
        contexts_train = np.memmap(os.path.join(data_dir, 'contexts_train.mmap'), dtype=np.uint16, mode='r',
                                   shape=(self.num_data_train, self.max_len))
        labels_train = np.memmap(os.path.join(data_dir, 'labels_train.mmap'), dtype=np.uint16, mode='r',
                                 shape=(self.num_data_train,))
        entities_x_valid = np.memmap(os.path.join(data_dir, 'entities_x_valid.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_valid,))
        entities_y_valid = np.memmap(os.path.join(data_dir, 'entities_y_valid.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_valid,))
        contexts_valid = np.memmap(os.path.join(data_dir, 'contexts_valid.mmap'), dtype=np.uint16, mode='r',
                                   shape=(self.num_data_valid, self.max_len))
        labels_valid = np.memmap(os.path.join(data_dir, 'labels_valid.mmap'), dtype=np.uint16, mode='r',
                                 shape=(self.num_data_valid,))
        entities_x_test = np.memmap(os.path.join(data_dir, 'entities_x_test.mmap'), dtype=np.uint16, mode='r',
                                    shape=(self.num_data_test,))
        entities_y_test = np.memmap(os.path.join(data_dir, 'entities_y_test.mmap'), dtype=np.uint16, mode='r',
                                    shape=(self.num_data_test,))
        contexts_test = np.memmap(os.path.join(data_dir, 'contexts_test.mmap'), dtype=np.uint16, mode='r',
                                  shape=(self.num_data_test, self.max_len))
        labels_test = np.memmap(os.path.join(data_dir, 'labels_test.mmap'), dtype=np.uint16, mode='r',
                                shape=(self.num_data_test,))

        return entities_x_train, entities_y_train, contexts_train, labels_train, entities_x_valid, entities_y_valid, \
            contexts_valid, labels_valid, entities_x_test, entities_y_test, contexts_test, labels_test

    def create_dataset(self, entities_x, entities_y, contexts, labels, n_batch):
        """
        Create the Tensorflow dataset to feed to the trainer.

        Parameters
        ----------
        entities_x : np.array
            The first entity in each sentence.
        entities_y : np.array
            The second entity in each sentence.
        contexts : np.array
            The contexts in which each entity pair occurs.
        labels : np.array
            The labels for each sentence.
        n_batch : int
            The batch size.

        Returns
        -------
        tf.data.Dataset
        """

        def gen():
            while True:
                inds = np.random.randint(0, len(entities_x), size=(n_batch,))

                x = np.int32(entities_x[inds])
                y = np.int32(entities_y[inds])
                c = np.int32(contexts[inds])
                r = np.int32(labels[inds])

                yield x, y, c, r

        return tf.data.Dataset.from_generator(gen, output_types=(tf.int32,)*4,
                                              output_shapes=([None], [None], [None, self.max_len], [None])
                                              )

    def train(self, n_iter, n_batch, n_samples, val_freq, n_batch_val, n_samples_val, save_freq=1000):
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
        val_freq : int
            The frequency (in iterations) at which to evaluate the performance on the validation set.
        n_batch_val : int
            The validation batch size.
        n_samples_val : int
            The number of latent samples to draw during validation.
        save_freq : int
            The frequency (in iterations) at which to save the parameter values.

        Returns
        -------
        """

        dataset_train = self.create_dataset(self.entities_x_train, self.entities_y_train, self.contexts_train,
                                            self.labels_train, n_batch)

        for (i, batch) in dataset_train.enumerate(start=1):
            start = time.perf_counter()

            x, y, c, r = batch

            obj = self.trainer.optimise(x, y, c, r, n_samples)

            print('Iteration ' + str(i.numpy()) + ': objective = ' + str(obj.numpy()) +
                  ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)')

            if i % val_freq == 0:
                self.test('valid', n_batch_val, n_samples_val)

            if i % save_freq == 0:
                self.checkpoint.write(os.path.join(self.out_dir, 'saved_model'))

            if i >= n_iter:
                break

    def test(self, valid_or_test, n_batch, n_samples, eval_at_optimal_threshold=False, min_threshold=0.5,
             max_threshold=0.99, num=50):
        """
        Evaluate the model performance on the either the validation or test set.

        Parameters
        ----------
        valid_or_test : str
            Takes the value either `valid` or `test`, depending on which dataset is to be evaluated.
        n_batch : int
            The number of data points to evaluate at a time.
        n_samples : int
            The number of latent samples to draw.
        eval_at_optimal_threshold : bool
            Whether or not to optimise the probability threshold at which predictions are made.
        min_threshold : float
            The smallest threshold to consider.
        max_threshold : float
            The largest threshold to consider.
        num : int
            The number of thresholds to evaluate.

        Returns
        -------
        """

        if valid_or_test == 'valid':
            entities_x = self.entities_x_valid
            entities_y = self.entities_y_valid
            contexts = self.contexts_valid
            labels = self.labels_valid

            num_data = self.num_data_valid
        else:
            entities_x = self.entities_x_test
            entities_y = self.entities_y_test
            contexts = self.contexts_test
            labels = self.labels_test

            num_data = self.num_data_test

        start = time.perf_counter()

        obj = 0

        log_probs = []
        r_pred = []

        for n in range(0, num_data, n_batch):
            x = tf.constant(np.int32(entities_x[n: n + n_batch]))
            y = tf.constant(np.int32(entities_y[n: n + n_batch]))
            c = tf.constant(np.int32(contexts[n: n + n_batch]))
            r = tf.constant(np.int32(labels[n: n + n_batch]))

            obj += self.trainer.objective(x, y, c, r, n_samples).numpy() * len(x)

            log_probs_n, r_pred_n = self.trainer.classify(x, y, c, n_samples)

            log_probs.append(log_probs_n.numpy())
            r_pred.append(r_pred_n.numpy())

        obj /= num_data

        log_probs = np.concatenate(log_probs)
        r_pred = np.concatenate(r_pred)

        np.save(os.path.join(self.out_dir, 'log_probs_' + valid_or_test + '.npy'), log_probs, allow_pickle=False)
        np.save(os.path.join(self.out_dir, 'predictions_' + valid_or_test + '.npy'), r_pred, allow_pickle=False)

        if eval_at_optimal_threshold:
            optimal_threshold = optimise_threshold(labels=labels, scores=log_probs, label_types=self.label_types,
                                                   neg_label=self.neg_label, min_threshold=min_threshold,
                                                   max_threshold=max_threshold, num=num)

            argmaxes = np.argmax(log_probs, axis=-1)
            maxes = log_probs[np.arange(len(argmaxes)), argmaxes]

            r_pred = [argmaxes[i] if maxes[i] >= optimal_threshold else self.label_types.index(self.neg_label)
                      for i in range(len(argmaxes))]

        precision, recall, f1 = precision_recall_f1(labels, r_pred, self.label_types, self.neg_label)

        print('Test: objective = ' + str(obj) + ' precision = ' + str(precision) + ' recall = ' + str(recall) +
              ' f1 = ' + str(f1) + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)')

        return obj, f1


class MaximumLikelihoodPairLevel(object):
    """
    This Run class handles initialising the trainer, loading data, creating the Tensorflow dataset and model training
    for pair-level classification.

    Parameters
    ----------
    data_dir : str
        The directory containing the data. It should contain the following 12 memory-mapped Numpy arrays:
            entities_x_{train,valid,test}.mmap: the entity types for the first entity.
            entities_y_{train,valid,test}.mmap: the entity types for the second entity.
            labels_{train,valid,test}.mmap: the label classifying the relation type between the two entities.
    vocab : list[str]
        The list of valid tokens.
    entity_types : list[str]
        The list of valid entity types (effectively the vocabulary for the entities).
    pos_entity_pairs : list[str]
        The list of entity pairs which have a relation other than no_relation. Each element of this list contains the
        two unique entity identifiers (elements of `entity_types`) joined together with a `:`.
    label_types : list[str]
        The list of valid label types.
    neg_label : str
        The label indicating that two entities are not related.
    num_data_train : int
        The number of training data points.
    num_data_valid : int
        The number of validation data points.
    num_data_test : int
        The number of test data points.
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

    def __init__(self, data_dir, vocab, entity_types, pos_entity_pairs, label_types, neg_label, num_data_train,
                 num_data_valid, num_data_test, max_len, trainer, trainer_kwargs, out_dir, pre_trained=False,
                 pre_trained_dir=None):

        self.vocab = vocab
        self.entity_types = entity_types
        self.pos_entity_pairs = set(pos_entity_pairs)
        self.label_types = label_types
        self.neg_label = neg_label

        self.num_data_train = num_data_train
        self.num_data_valid = num_data_valid
        self.num_data_test = num_data_test
        self.max_len = max_len

        self.trainer = trainer(**trainer_kwargs)

        self.entities_x_train, self.entities_y_train, self.labels_train, self.entities_x_valid, self.entities_y_valid, \
            self.labels_valid, self.entities_x_test, self.entities_y_test, self.labels_test = self.load_data(data_dir)

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
            The directory containing the data. It should contain the following 12 memory-mapped Numpy arrays:
                entities_x_{train,valid,test}.mmap: the entity types for the first entity.
                entities_y_{train,valid,test}.mmap: the entity types for the second entity.
                labels_{train,valid,test}.mmap: the label classifying the relation type between the two entities.

        Returns
        -------
        entities_x_train : np.memmap
            The indices to `self.entity_types` for the first entity in the training data.
        entities_y_train : np.memmap
            The indices to `self.entity_types` for the second entity in the training data.
        labels_train : np.memmap
            The indices to `self.label_types` for the labels for each entity pair in the training data.
        entities_x_valid : np.memmap
            The indices to `self.entity_types` for the first entity in the validation data.
        entities_y_valid : np.memmap
            The indices to `self.entity_types` for the second entity in the validation data.
        labels_valid : np.memmap
            The indices to `self.label_types` for the labels for each entity pair in the validation data.
        entities_x_test : np.memmap
            The indices to `self.entity_types` for the first entity in the test data.
        entities_y_test : np.memmap
            The indices to `self.entity_types` for the second entity in the test data.
        labels_test : np.memmap
            The indices to `self.label_types` for the labels for each entity pair in the test data.
        """

        entities_x_train = np.memmap(os.path.join(data_dir, 'entities_x_train.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_train,))
        entities_y_train = np.memmap(os.path.join(data_dir, 'entities_y_train.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_train,))
        labels_train = np.memmap(os.path.join(data_dir, 'labels_train.mmap'), dtype=np.uint16, mode='r',
                                 shape=(self.num_data_train,))
        entities_x_valid = np.memmap(os.path.join(data_dir, 'entities_x_valid.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_valid,))
        entities_y_valid = np.memmap(os.path.join(data_dir, 'entities_y_valid.mmap'), dtype=np.uint16, mode='r',
                                     shape=(self.num_data_valid,))
        labels_valid = np.memmap(os.path.join(data_dir, 'labels_valid.mmap'), dtype=np.uint16, mode='r',
                                 shape=(self.num_data_valid,))
        entities_x_test = np.memmap(os.path.join(data_dir, 'entities_x_test.mmap'), dtype=np.uint16, mode='r',
                                    shape=(self.num_data_test,))
        entities_y_test = np.memmap(os.path.join(data_dir, 'entities_y_test.mmap'), dtype=np.uint16, mode='r',
                                    shape=(self.num_data_test,))
        labels_test = np.memmap(os.path.join(data_dir, 'labels_test.mmap'), dtype=np.uint16, mode='r',
                                shape=(self.num_data_test,))

        return entities_x_train, entities_y_train, labels_train, entities_x_valid, entities_y_valid, labels_valid, \
            entities_x_test, entities_y_test, labels_test

    def create_dataset(self, entities_x, entities_y, labels, n_batch, n_neg):
        """
        Create the Tensorflow dataset to feed to the trainer.

        Parameters
        ----------
        entities_x : np.array
            The first entity for each data point.
        entities_y : np.array
            The second entity for each data point.
        labels : np.array
            The labels for each data point.
        n_batch : int
            The batch size, excluding negative examples.
        n_neg : int
            The number of negative (i.e. no_relation) examples to include in the batch.

        Returns
        -------
        tf.data.Dataset
        """

        def gen():
            entity_pairs_valid = {':'.join([str(self.entities_x_valid[n]), str(self.entities_y_valid[n])])
                                  for n in range(self.num_data_valid)}
            entity_pairs_test = {':'.join([str(self.entities_x_test[n]), str(self.entities_y_test[n])])
                                 for n in range(self.num_data_test)}

            while True:
                inds = np.random.randint(0, len(entities_x), size=(n_batch - n_neg,))

                x = np.int32(entities_x[inds])
                y = np.int32(entities_y[inds])
                r = np.int32(labels[inds])

                x_neg = []
                y_neg = []

                while len(x_neg) < n_neg:
                    cand_x = np.random.randint(0, len(self.entity_types))
                    cand_y = np.random.randint(0, len(self.entity_types))

                    if (':'.join([self.entity_types[cand_x], self.entity_types[cand_y]]) not in self.pos_entity_pairs
                        and
                        ':'.join([self.entity_types[cand_y], self.entity_types[cand_x]]) not in self.pos_entity_pairs
                        and
                        ':'.join([str(cand_x), str(cand_y)]) not in entity_pairs_valid
                        and
                        ':'.join([str(cand_y), str(cand_x)]) not in entity_pairs_valid
                        and
                        ':'.join([str(cand_x), str(cand_y)]) not in entity_pairs_test
                        and
                        ':'.join([str(cand_y), str(cand_x)]) not in entity_pairs_test):
                        x_neg.append(cand_x)
                        y_neg.append(cand_y)

                x_neg = np.array(x_neg)
                y_neg = np.array(y_neg)
                r_neg = self.label_types.index(self.neg_label) * np.ones((n_neg,), dtype=np.int32)

                x = np.concatenate([x, x_neg])
                y = np.concatenate([y, y_neg])
                r = np.concatenate([r, r_neg])

                yield x, y, r

        return tf.data.Dataset.from_generator(gen, output_types=(tf.int32,)*3, output_shapes=([None], [None], [None]))

    def train(self, n_iter, n_batch, n_neg, n_samples, val_freq, n_batch_val, n_samples_val, save_freq=1000):
        """
        Train the model, saving the parameters at a fixed frequency using Tensorflow checkpoints.

        Parameters
        ----------
        n_iter : int
            The number of iterations for which to train the model.
        n_batch : int
            The training batch size, excluding negative examples.
        n_neg : int
            The number of negative (i.e. no_relation) examples to include in each batch.
        n_samples : int
            The number of latent samples to draw during training.
        val_freq : int
            The frequency (in iterations) at which to evaluate the performance on the validation set.
        n_batch_val : int
            The validation batch size.
        n_samples_val : int
            The number of latent samples to draw during validation.
        save_freq : int
            The frequency (in iterations) at which to save the parameter values.

        Returns
        -------
        """

        dataset_train = self.create_dataset(self.entities_x_train, self.entities_y_train, self.labels_train, n_batch,
                                            n_neg)

        for (i, batch) in dataset_train.enumerate(start=1):
            start = time.perf_counter()

            x, y, r = batch

            obj = self.trainer.optimise(x, y, r, n_samples)

            print('Iteration ' + str(i.numpy()) + ': objective = ' + str(obj.numpy()) +
                  ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)')

            if i % val_freq == 0:
                self.test('valid', n_batch_val, n_samples_val)

            if i % save_freq == 0:
                self.checkpoint.write(os.path.join(self.out_dir, 'saved_model'))

            if i >= n_iter:
                break

    def test(self, valid_or_test, n_batch, n_samples, eval_at_optimal_threshold=False, min_threshold=0.5,
             max_threshold=0.99, num=50):
        """
        Evaluate the model performance on the either the validation or test set.

        Parameters
        ----------
        valid_or_test : str
            Takes the value either `valid` or `test`, depending on which dataset is to be evaluated.
        n_batch : int
            The number of data points to evaluate at a time.
        n_samples : int
            The number of latent samples to draw.
        eval_at_optimal_threshold : bool
            Whether or not to optimise the probability threshold at which predictions are made.
        min_threshold : float
            The smallest threshold to consider.
        max_threshold : float
            The largest threshold to consider.
        num : int
            The number of thresholds to evaluate.

        Returns
        -------
        obj : float
            The training objective
        f1 : float
            The prediction f1 score
        """

        if valid_or_test == 'valid':
            entities_x = self.entities_x_valid
            entities_y = self.entities_y_valid
            labels = self.labels_valid

            num_data = self.num_data_valid
        else:
            entities_x = self.entities_x_test
            entities_y = self.entities_y_test
            labels = self.labels_test

            num_data = self.num_data_test

        start = time.perf_counter()

        obj = 0

        log_probs = []
        r_pred = []

        for n in range(0, num_data, n_batch):
            x = tf.constant(np.int32(entities_x[n: n + n_batch]))
            y = tf.constant(np.int32(entities_y[n: n + n_batch]))
            r = tf.constant(np.int32(labels[n: n + n_batch]))

            obj += self.trainer.objective(x, y, r, n_samples).numpy() * len(x)

            log_probs_n, r_pred_n = self.trainer.classify(x, y, n_samples)

            log_probs.append(log_probs_n.numpy())
            r_pred.append(r_pred_n.numpy())

        obj /= num_data

        log_probs = np.concatenate(log_probs)
        r_pred = np.concatenate(r_pred)

        np.save(os.path.join(self.out_dir, 'log_probs_' + valid_or_test + '.npy'), log_probs, allow_pickle=False)
        np.save(os.path.join(self.out_dir, 'predictions_' + valid_or_test + '.npy'), r_pred, allow_pickle=False)

        if eval_at_optimal_threshold:
            optimal_threshold = optimise_threshold(labels=labels, scores=log_probs, label_types=self.label_types,
                                                   neg_label=self.neg_label, min_threshold=min_threshold,
                                                   max_threshold=max_threshold, num=num)

            argmaxes = np.argmax(log_probs, axis=-1)
            maxes = log_probs[np.arange(len(argmaxes)), argmaxes]

            r_pred = [argmaxes[i] if maxes[i] >= optimal_threshold else self.label_types.index(self.neg_label)
                      for i in range(len(argmaxes))]

        precision, recall, f1 = precision_recall_f1(labels, r_pred, self.label_types, self.neg_label)

        print('Test: objective = ' + str(obj) + ' precision = ' + str(precision) + ' recall = ' + str(recall) +
              ' f1 = ' + str(f1) + ' (time taken = ' + str(time.perf_counter() - start) + ' seconds)')

        return obj, f1
