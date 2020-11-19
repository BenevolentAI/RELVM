from models.unsup.generative import ConditionalInfiniteMixtureLSTMAutoregressive as GenModelUnsup
from models.unsup.recognition import GaussianBiLSTM as RecModelUnsup
from trainers.unsup import SGVB as TrainerUnsup

from models.sup.classification import Classification as Model
from trainers.sup import MaximumLikelihoodPairLevel as Trainer
from run.sup import MaximumLikelihoodPairLevel as Run

import os
import json
import sys
import numpy as np
import tensorflow as tf


out_dir = sys.argv[1]


trainer_unsup_pre_trained_dir = 'exp_outputs/unsup/'


pre_trained = False
pre_trained_dir = ''


unsup_data_dir = 'data/unsupervised/'

data_dir = 'data/supervised/pair_level/'

with open(os.path.join(unsup_data_dir, 'vocab.json'), 'r') as f:
    vocab = json.loads(f.read())

with open(os.path.join(unsup_data_dir, 'entity_types.json'), 'r') as f:
    entity_types = json.loads(f.read())

with open(os.path.join(data_dir, 'pos_entity_pairs.json'), 'r') as f:
    pos_entity_pairs = json.loads(f.read())

with open(os.path.join(data_dir, 'label_types.json'), 'r') as f:
    label_types = json.loads(f.read())


num_data_train = 10
num_data_valid = 10
num_data_test = 10

n_classes = len(label_types)
neg_label = 'no_relation'

max_len = 140
emb_dim = 300
u_dim = 300
z_dim = 300

emb_matrix_entities = np.float32(np.random.normal(scale=0.1, size=(len(entity_types), emb_dim)))
emb_matrix_words = np.float32(np.random.normal(scale=0.1, size=(len(vocab), emb_dim)))

pad_ind = vocab.index('<PAD>')

optimiser_unsup = tf.keras.optimizers.Adam
optimiser_unsup_kwargs = {'learning_rate': 0.0001}

gen_nn_z_unsup_kwargs = {
    'ff_depth': 2,
    'ff_units': 1024,
    'ff_activation': 'relu'
}

gen_nn_context_unsup_kwargs = {
    'ff_depth': 2,
    'ff_units': 1024,
    'ff_activation': 'relu',
    'lstm_depth': 1,
    'lstm_units': 1024,
    'token_drop': 0.5,
}

gen_model_unsup_kwargs = {
    'max_len': max_len,
    'emb_dim': emb_dim,
    'u_dim': u_dim,
    'z_dim': z_dim,
    'nn_z_kwargs': gen_nn_z_unsup_kwargs,
    'nn_context_kwargs': gen_nn_context_unsup_kwargs,
}

rec_nn_unsup_kwargs = {
    'lstm_depth': 1,
    'lstm_units': 1024,
    'ff_depth': 2,
    'ff_units': 1024,
    'ff_activation': 'relu',
}

rec_model_unsup_kwargs = {
    'max_len': max_len,
    'emb_dim': emb_dim,
    'u_dim': u_dim,
    'nn_kwargs': rec_nn_unsup_kwargs,
}

trainer_unsup_kwargs = {
    'strategy': None,
    'optimiser': optimiser_unsup,
    'optimiser_kwargs': optimiser_unsup_kwargs,
    'emb_matrix_entities': emb_matrix_entities,
    'emb_matrix_words': emb_matrix_words,
    'pad_ind': pad_ind,
    'gen_model': GenModelUnsup,
    'gen_model_kwargs': gen_model_unsup_kwargs,
    'rec_model': RecModelUnsup,
    'rec_model_kwargs': rec_model_unsup_kwargs,
}


optimiser = tf.keras.optimizers.Adam
optimiser_kwargs = {'learning_rate': 0.0001}

model_nn_kwargs = {
    'ff_depth': 1,
    'ff_units': 300,
    'ff_activation': 'relu',
}

model_kwargs = {
    'z_dim': z_dim,
    'n_classes': n_classes,
    'nn_kwargs': model_nn_kwargs,
}

trainer_kwargs = {
    'optimiser': optimiser,
    'optimiser_kwargs': optimiser_kwargs,
    'trainer_unsup': TrainerUnsup,
    'trainer_unsup_kwargs': trainer_unsup_kwargs,
    'trainer_unsup_pre_trained_dir': trainer_unsup_pre_trained_dir,
    'model': Model,
    'model_kwargs': model_kwargs,
    'fine_tune_unsup': False,
}


train = True
n_iter_train = 100000
n_batch_train = 512
n_neg_train = 448
n_samples_train = 4

val_freq = 1000
n_batch_val = 128
n_samples_val = 8


test = True
n_batch_test = 128
n_samples_test = 8


if __name__ == '__main__':

    run = Run(data_dir=data_dir, vocab=vocab, entity_types=entity_types, pos_entity_pairs=pos_entity_pairs,
              label_types=label_types, neg_label=neg_label, num_data_train=num_data_train,
              num_data_valid=num_data_valid, num_data_test=num_data_test, max_len=max_len, trainer=Trainer,
              trainer_kwargs=trainer_kwargs, out_dir=out_dir, pre_trained=pre_trained, pre_trained_dir=pre_trained_dir)

    if train:
        run.train(n_iter=n_iter_train, n_batch=n_batch_train, n_neg=n_neg_train, n_samples=n_samples_train,
                  val_freq=val_freq, n_batch_val=n_batch_val, n_samples_val=n_samples_val)

    if test:
        run.test('test', n_batch=n_batch_test, n_samples=n_samples_test, eval_at_optimal_threshold=True,
                 min_threshold=0.99, max_threshold=0.9995, num=20)
