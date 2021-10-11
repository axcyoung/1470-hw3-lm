import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = 20 # DO NOT CHANGE!
        self.embedding_size = 128
        self.batch_size = 64

        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=.1, dtype=tf.float32))
        self.layer1 = tf.keras.layers.GRU(256, activation='relu', return_sequences=True, return_state=True) #ACTIVATION?
        self.layer2 = tf.keras.layers.Dense(4096, activation='relu')
        self.layer3 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        layer1Output, final_state = self.layer1(embedding, initial_state=initial_state)
        layer2Output = self.layer2(layer1Output)
        layer3Output = self.layer3(layer2Output)
        probs = tf.nn.softmax(layer3Output)
        return probs, final_state

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    
    train_inputs = train_inputs[:train_inputs.shape[0] // 20 * 20]
    train_labels = train_labels[:train_labels.shape[0] // 20 * 20]
    train_inputs = tf.reshape(train_inputs, [-1, model.window_size])
    train_labels = tf.reshape(train_labels, [-1, model.window_size])
    
    num_examples = train_labels.shape.as_list()[0]
    shuffle_indices = np.arange(0, num_examples)
    shuffle_indices = tf.random.shuffle(shuffle_indices)
    train_inputs = tf.gather(train_inputs, shuffle_indices)
    train_labels = tf.gather(train_labels, shuffle_indices)

    optimizer = tf.keras.optimizers.Adam(.001)

    for i in range(0, num_examples, model.batch_size):
        input_batch = train_inputs[i:i + model.batch_size, :]
        label_batch = train_labels[i:i + model.batch_size, :]
        
        with tf.GradientTape() as tape:
            probs = model.call(input_batch, None)
            probs = probs[0]
            loss = model.loss(probs, label_batch)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    test_inputs = test_inputs[:test_inputs.shape[0] // 20 * 20]
    test_labels = test_labels[:test_labels.shape[0] // 20 * 20]
    test_inputs = tf.reshape(test_inputs, [-1, model.window_size])
    test_labels = tf.reshape(test_labels, [-1, model.window_size])
    
    perplexities = []
    for i in range(0, test_labels.shape.as_list()[0], model.batch_size):
        input_batch = test_inputs[i:i + model.batch_size, :]
        label_batch = test_labels[i:i + model.batch_size, :]
        
        probs = model.call(input_batch, None)
        probs = probs[0]
        perplexities.append(model.loss(probs, label_batch))
    return np.exp(np.mean(np.array(perplexities)))


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    #NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    all_data = get_data('data/train.txt', 'data/test.txt')
    trainid = all_data[0]
    testid = all_data[1]
    word2id = all_data[2]
    
    #trainid = trainid[:len(trainid) // 20 * 20 + 1]
    #testid = testid[:len(testid) // 20 * 20 + 1]
    
    train_inputs = []
    train_labels = []
    for i in range(0, len(trainid)-1, 20):
        train_inputs += (trainid[i:i+20])
        train_labels += (trainid[i+1:i+21])
    test_inputs = []
    test_labels = []
    for i in range(0, len(testid)-1, 20):
        test_inputs += (testid[i:i+20])
        test_labels += (testid[i+1:i+21])
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)

    model = Model(len(word2id))
    
    train(model, train_inputs, train_labels)
    
    perplexity = test(model, test_inputs, test_labels)
    
    print(perplexity)

    generate_sentence('the', 'west', 8, word2id, model)

if __name__ == '__main__':
    main()
