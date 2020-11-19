# RELVM

This repository contains the code accompanying the paper *"Learning Informative Representations of Biomedical Relations with Latent Variable Models", Harshil Shah and Julien Fauqueur, EMNLP SustaiNLP 2020*.

## Requirements

- Python 3.7
  - Numpy >= 1.17.2
  - Tensorflow >= 2.0.0

## Instructions

### Introduction

The code in this repository is for training a latent variable generative model of pairs of entities and the contexts (i.e. sentences) in which the entities occur. The representations from this model can then be used to perform both mention-level and pair-level classification.

Throughout the code, the following conventions are used:

- `x` or `entities_x` will refer to the first entity in a context.
- `y` or `entities_y` will refer to the second entity in a context.
- `c` or `contexts` will refer to the context (i.e. sentence) in which the entities occur.
- `r` or `labels` will refer to the class label when performing either mention-level or pair-level classification.



### Data

To avoid out-of-memory issues, the data is stored in [memory-mapped Numpy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html). The metadata is stored in JSON files.

#### Unsupervised

The unsupervised data directory (e.g. `data/unsupervised`) should contain the following JSON files (which contain the metadata):

- `vocab.json`
  - This is a list of strings. It is the set of possible tokens which can appear in the contexts.
- `entity_types.json`
  - This is a list of strings. It is the set of possible values for the entities. If training the model for mention-level classification, this will be a list of entity types. If training the model for pair-level classification, this will be a list of entity identifiers.

It should also contain the following memory-mapped Numpy arrays:

- `entities_x.mmap`
  - This is a one-dimensional array. The i<sup>th</sup> row contains the index to `entity_types.json` for the first entity in the i<sup>th</sup> context.
- `entities_y.mmap`
  - This is a one-dimensional array. The i<sup>th</sup> row contains the index to `entity_types.json` for the second entity in the i<sup>th</sup> context.
- `contexts.mmap`
  - This is a two-dimensional array. The i<sup>th</sup> row contains the indices to `vocab.json` for the i<sup>th</sup> context.

#### Mention-level classification

The mention-level classification data directory (e.g. `data/supervised/mention_level`) should contain the following JSON files (which contain the metadata):

- `label_types.json`
  - This is a list of strings. It is the set of possible values for the labels.

It should also contain the following memory-mapped Numpy arrays (for the training, validation, and test data respectively):

- `entities_x_{train,valid,test}.mmap`
  - These are one-dimensional arrays. The i<sup>th</sup> row contains the index to `entity_types.json` from the unsupervised data directory for the first entity in the i<sup>th</sup> context.
- `entities_y_{train,valid,test}.mmap`
  - These are one-dimensional arrays. The i<sup>th</sup> row contains the index to `entity_types.json` from the unsupervised data directory for the second entity in the i<sup>th</sup> context.
- `contexts_{train,valid,test}.mmap`
  - These are two-dimensional arrays. The i<sup>th</sup> row contains the indices to `vocab.json` from the unsupervised data directory for the i<sup>th</sup> context.
- `labels_{train,valid,test}.mmap`
  - These are one-dimensional arrays. The i<sup>th</sup> row contains the index to `label_types.json` for the i<sup>th</sup> entity pair and context.

#### Pair-level classification

The pair-level classification data directory (e.g. `data/supervised/pair_level`) should contain the following JSON files (which contain the metadata):

- `label_types.json`
  - This is a list of strings. It is the set of possible values for the labels.
- `pos_entity_pairs.json`
  - This is a list of strings containing the entity pairs which have a positive relation. Each element of this list contains the unique entity identifiers for the pair, joined together with a `:`.

It should also contain the following memory-mapped Numpy arrays (for the training, validation, and test data respectively):

- `entities_x_{train,valid,test}.mmap`
  - These are one-dimensional arrays. The i<sup>th</sup> row contains the index to `entity_types.json` from the unsupervised data directory for the first entity in the i<sup>th</sup> pair.
- `entities_y_{train,valid,test}.mmap`
  - These are one-dimensional arrays. The i<sup>th</sup> row contains the index to `entity_types.json` from the unsupervised data directory for the second entity in the i<sup>th</sup> pair.
- `labels_{train,valid,test}.mmap`
  - These are one-dimensional arrays. The i<sup>th</sup> row contains the index to `label_types.json` for the i<sup>th</sup> entity pair.

### Training and evaluating

#### Unsupervised representation learning

To train the unsupervised representation model, run `exp_unsup.py`, specifying the directory in which to store the model parameters. For example:

```
mkdir -p exp_outputs/unsup

python3 exp_unsup.py exp_outputs/unsup
```

#### Mention-level classification

Once the unsupervised representation model has been trained, the mention-level classification model can be trained and evaluated by running `exp_classification_mention.py`. First, the following variables in `exp_classification_mention.py` must be set: 

- `trainer_unsup_pre_trained_dir` must be set to the directory with the saved parameters from the unsupervised representation model (e.g. `exp_outputs/unsup`).
- `unsup_data_dir` must be set to the directory with the data used to train the unsupervised representation model (e.g. `data/unsupervised`).

When running `exp_classification_mention.py`, the directory in which to store the classification model parameters must be specified. For example:

```
mkdir -p exp_outputs/classification_mention

python3 exp_classification_mention.py exp_outputs/classification_mention
```

#### Pair-level classification

The pair-level classification model can be trained and evaluated by running `exp_classification_pair.py` in an identical fashion to the mention-level classification model. Again, the following variables in `exp_classification_pair.py` must be set: 

- `trainer_unsup_pre_trained_dir` must be set to the directory with the saved parameters from the unsupervised representation model (e.g. `exp_outputs/unsup`).
- `unsup_data_dir` must be set to the directory with the data used to train the unsupervised representation model (e.g. `data/unsupervised`).

When running `exp_classification_pair.py`, the directory in which to store the classification model parameters must be specified. For example:

```
mkdir -p exp_outputs/classification_pair

python3 exp_classification_pair.py exp_outputs/classification_pair
```

### Component testing
From the project root folder, the following 3 scripts should be run in this order and all return `OK`. 
```
python3 tests/test_unsup.py
python3 tests/test_classification_mention.py
python3 tests/test_classification_pair.py
```
