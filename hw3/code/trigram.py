import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from functools import reduce
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        self.vocab_size = vocab_size #len(word2id)
        self.embedding_size = 32
        self.batch_size = 1024

        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.W = tf.Variable(tf.random.truncated_normal([self.embedding_size*2, self.vocab_size], stddev=.1, dtype=tf.float32))
        self.b = tf.Variable(tf.random.truncated_normal([self.vocab_size], stddev=.1, dtype=tf.float32))

    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)
        :return: probs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        embedding = tf.nn.embedding_lookup(self.E, inputs)
        print(embedding.shape)
        embedding = tf.reshape(embedding, [embedding.shape.as_list()[0], self.embedding_size*2])
        logits = tf.matmul(embedding, self.W) + self.b
        probabilities = tf.nn.softmax(logits)
        return probabilities

    def loss_function(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: Please use np.reduce_mean and not np.reduce_sum when calculating your loss.
        
        :param probs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))


def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples. 
    You should take the train input and shape them into groups of two words.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order. 
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    
    num_examples = train_labels.shape[0]
    shuffle_indices = np.arange(0, num_examples)
    shuffle_indices = tf.random.shuffle(shuffle_indices)
    train_input = tf.gather(train_input, shuffle_indices)
    train_labels = tf.gather(train_labels, shuffle_indices)

    optimizer = tf.keras.optimizers.Adam(.001)

    for i in range(0, num_examples, model.batch_size):
        input_batch = train_input[i:i + model.batch_size, :]
        label_batch = train_labels[i:i + model.batch_size]
        
        with tf.GradientTape() as tape:
            probs = model.call(input_batch)
            loss = model.loss_function(probs, label_batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        

def test(model, test_input, test_labels):
    """
    Runs through all test examples. You should take the test input and shape them into groups of two words.
    And test input should be batched here as well.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: perplexity of the test set
    """
    
    perplexities = []
    for i in range(0, test_labels.shape[0], model.batch_size):
        input_batch = test_input[i:i + model.batch_size, :]
        label_batch = test_labels[i:i + model.batch_size]
        
        probs = model.call(input_batch)
        perplexities.append(model.loss_function(probs, label_batch))
    return np.exp(np.mean(np.array(perplexities)))


def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of targeted length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """

    #NOTE: This is a deterministic, argmax sentence generation

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(
            model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))


def main():
    
    all_data = get_data('data/train.txt', 'data/test.txt')
    trainid = all_data[0]
    testid = all_data[1]
    word2id = all_data[2]
    
    train_input = []
    train_labels = []
    for i in range(0, len(trainid)-2):
        train_input.append([trainid[i], trainid[i+1]])
        train_labels.append(trainid[i+2])
    test_input = []
    test_labels = []
    for i in range(0, len(testid)-2):
        test_input.append([testid[i], testid[i+1]])
        test_labels.append(testid[i+2])
    train_input = np.array(train_input)
    train_labels = np.array(train_labels)
    test_input = np.array(test_input)
    test_labels = np.array(test_labels)

    model = Model(len(word2id))
    
    train(model, train_input, train_labels)
    
    perplexity = test(model, test_input, test_labels)
    
    print(perplexity)

    generate_sentence('the', 'west', 8, word2id, model)

if __name__ == '__main__':
    main()
