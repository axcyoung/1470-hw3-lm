import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """
    # TODO: load and concatenate training data from training file.

    # TODO: load and concatenate testing data from testing file.

    # TODO: read in and tokenize training data

    # TODO: read in and tokenize testing data

    # BONUS: Ensure that all words appearing in test also appear in train

    # TODO: return tuple of training tokens, testing tokens, and the vocab dictionary.

    pass
