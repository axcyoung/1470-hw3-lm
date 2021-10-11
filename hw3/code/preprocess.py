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

    train = []
    with open(train_file, 'r') as tr:
        for line in tr:
            train += line.strip().split()
    test = []
    with open(test_file, 'r') as te:
        for line in te:
             test += line.strip().split()

    vocab = set(train) # collects all unique words in our dataset (vocab)
    word2id = {w: i for i, w in enumerate(list(vocab))} # maps each word in our vocab to a unique index
    
    trainid = []
    for token in train:
        trainid += [word2id[token]]
    testid = []
    for token in test:
        testid += [word2id[token]]
    print(np.shape(np.array(trainid)))
    print(np.shape(np.array(test)))
    return (trainid, testid, word2id)