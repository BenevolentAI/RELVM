from models.unsup.generative \
    import ConditionalInfiniteMixtureLSTMAutoregressive as GenModel
from models.unsup.recognition import GaussianBiLSTM as RecModel
from trainers.unsup import SGVB as Trainer
from run.unsup import Run

import json
import os
import numpy as np
import tensorflow as tf
import string
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = 'tests/data/unsupervised/'
out_dir = 'tests/exp_outputs/unsup/'


pre_trained = False
pre_trained_dir = ''


if not os.path.exists(os.path.join(data_dir, 'vocab.json')):
    vocab = ['<PAD>', '<ENT>', '<UNK>'] + \
            [''.join(random.choice(string.ascii_letters)
                     for i in range(10)) for j in range(60000)]
    with open(os.path.join(data_dir, 'vocab.json'), 'w') as jsonfile:
        json.dump(vocab, jsonfile)

with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
    vocab = json.loads(f.read())


with open(os.path.join(data_dir, 'entity_types.json'), 'r') as f:
    entity_types = json.loads(f.read())


num_data = 10

max_len = 140
emb_dim = 300
u_dim = 300
z_dim = 300

emb_matrix_entities = np.float32(
    np.random.normal(scale=0.1, size=(len(entity_types), emb_dim)))
emb_matrix_words = np.float32(
    np.random.normal(scale=0.1, size=(len(vocab), emb_dim)))

pad_ind = vocab.index('<PAD>')


optimiser = tf.keras.optimizers.Adam
optimiser_kwargs = {'learning_rate': 0.0001}

gen_nn_z_kwargs = {
    'ff_depth': 2,
    'ff_units': 16,  # 1024,
    'ff_activation': 'relu',
}

gen_nn_context_kwargs = {
    'ff_depth': 2,
    'ff_units': 16,  # 1024,
    'ff_activation': 'relu',
    'lstm_depth': 1,
    'lstm_units': 16,  # 1024,
    'token_drop': 0.5,
}

gen_model_kwargs = {
    'max_len': max_len,
    'emb_dim': emb_dim,
    'u_dim': u_dim,
    'z_dim': z_dim,
    'nn_z_kwargs': gen_nn_z_kwargs,
    'nn_context_kwargs': gen_nn_context_kwargs,
}

rec_nn_kwargs = {
    'lstm_depth': 1,
    'lstm_units': 16,  # 1024,
    'ff_depth': 2,
    'ff_units': 16,  # 1024,
    'ff_activation': 'relu',
}

rec_model_kwargs = {
    'max_len': max_len,
    'emb_dim': emb_dim,
    'u_dim': u_dim,
    'nn_kwargs': rec_nn_kwargs,
}

trainer_kwargs = {
    'optimiser': optimiser,
    'optimiser_kwargs': optimiser_kwargs,
    'emb_matrix_entities': emb_matrix_entities,
    'emb_matrix_words': emb_matrix_words,
    'pad_ind': pad_ind,
    'gen_model': GenModel,
    'gen_model_kwargs': gen_model_kwargs,
    'rec_model': RecModel,
    'rec_model_kwargs': rec_model_kwargs,
}

n_iter_train = 3  # 400000
n_batch_train = 1  # 192
n_samples_train = 10
n_iter_warm_up_train = 1  # 10000

expected_elbo_kl = [-132.07957458496094, 0.00006896428]
mmap_files_to_delete = []


class UnsupTest (tf.test.TestCase):

    def make_memmaps(self):
        """
        Make the memory-mapped Numpy arrays which will store the final data.

        entities_x : np.memmap
            The first entity in a sentence.
            The first column contains the index to the UUIDS;
            the second contains the index
            to the entity type.
        entities_y : np.memmap
            The second entity in a sentence.
            The first column contains the index to the UUIDS;
            the second contains the index
            to the entity type.
        contexts : np.memmap
            The indices to the processing vocabulary for the context of
            each sentence.
        """

        def create_mmap(filename, n_indices, num_data):
            full_path = os.path.join(data_dir, filename)
            mm = np.memmap(full_path,
                           dtype=np.uint16, mode='w+',
                           shape=(num_data, 1))
            mm[:, 0] = np.random.randint(0, n_indices, num_data)
            mmap_files_to_delete.append(full_path)

        def create_context_mmap(filename, n_indices, num_data):
            full_path = os.path.join(data_dir, filename)
            mm = np.memmap(full_path,
                           dtype=np.uint16, mode='w+',
                           shape=(num_data, max_len))
            mm[:] = np.random.randint(0, n_indices, size=(num_data, max_len))
            mmap_files_to_delete.append(full_path)

        create_mmap('entities_x.mmap', len(entity_types), num_data)
        create_mmap('entities_y.mmap', len(entity_types), num_data)
        create_context_mmap('contexts.mmap', len(vocab), num_data)

    def delete_memmaps_unsup(self):
        """
        Deletes the mock memmap data created above for the test
        """
        for f in mmap_files_to_delete:
            os.remove(f)

    def setUp(self):

        self.make_memmaps()

        run = Run(data_dir=data_dir, vocab=vocab, entity_types=entity_types,
                  num_data=num_data, max_len=max_len, trainer=Trainer,
                  trainer_kwargs=trainer_kwargs, out_dir=out_dir,
                  pre_trained=pre_trained, pre_trained_dir=pre_trained_dir)

        elbo_kl = run.train(n_iter=n_iter_train, n_batch=n_batch_train,
                            n_samples=n_samples_train,
                            n_iter_warm_up=n_iter_warm_up_train)

        last_row_to_test = np.shape(elbo_kl)[0]-1

        self.assertAlmostEqual(
            elbo_kl[last_row_to_test, 0], expected_elbo_kl[0], places=5)

        self.assertAlmostEqual(
            elbo_kl[last_row_to_test, 1], expected_elbo_kl[1], places=5)

        self.delete_memmaps_unsup()


tf.test.main()
