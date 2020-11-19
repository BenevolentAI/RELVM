from models.unsup.generative import ConditionalInfiniteMixtureLSTMAutoregressive as GenModel
from models.unsup.recognition import GaussianBiLSTM as RecModel
from trainers.unsup import SGVB as Trainer
from run.unsup import Run

import os
import sys
import json
import numpy as np
import tensorflow as tf


out_dir = sys.argv[1]


pre_trained = False
pre_trained_dir = ''


data_dir = 'data/unsupervised/'

with open(os.path.join(data_dir, 'vocab.json'), 'r') as f:
    vocab = json.loads(f.read())

with open(os.path.join(data_dir, 'entity_types.json'), 'r') as f:
    entity_types = json.loads(f.read())


num_data = 10

max_len = 140
emb_dim = 300
u_dim = 300
z_dim = 300

emb_matrix_entities = np.float32(np.random.normal(scale=0.1, size=(len(entity_types), emb_dim)))
emb_matrix_words = np.float32(np.random.normal(scale=0.1, size=(len(vocab), emb_dim)))

pad_ind = vocab.index('<PAD>')


optimiser = tf.keras.optimizers.Adam
optimiser_kwargs = {'learning_rate': 0.0001}

gen_nn_z_kwargs = {
    'ff_depth': 2,
    'ff_units': 1024,
    'ff_activation': 'relu',
}

gen_nn_context_kwargs = {
    'ff_depth': 2,
    'ff_units': 1024,
    'ff_activation': 'relu',
    'lstm_depth': 1,
    'lstm_units': 1024,
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
    'lstm_units': 1024,
    'ff_depth': 2,
    'ff_units': 1024,
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


train = True
n_iter_train = 400000
n_batch_train = 192
n_samples_train = 1
n_iter_warm_up_train = 10000


if __name__ == '__main__':

    run = Run(data_dir=data_dir, vocab=vocab, entity_types=entity_types, num_data=num_data, max_len=max_len,
              trainer=Trainer, trainer_kwargs=trainer_kwargs, out_dir=out_dir, pre_trained=pre_trained,
              pre_trained_dir=pre_trained_dir)

    if train:
        run.train(n_iter=n_iter_train, n_batch=n_batch_train, n_samples=n_samples_train,
                  n_iter_warm_up=n_iter_warm_up_train)
