import numpy
import random
import sys
import attr
import string
import re
import matplotlib.pyplot as plt

from keras.callbacks import Callback


def plot_loss(history):
    """
    Plots Keras history
    :param history:
    :return:
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('categorical cross entropy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def create_indices(df, column):
    """
    Creates look up tables for tokens from a corpus.

    token_to_index, index_to_token = create_indices(df, 'name')

    # Look up the index of the token 'a'
    index_of_a = token_to_index['a']

    token = index_to_token[0]

    :param df: The DataFrame to use to build up the indices
    :param column: The string name of the column holding the text
    :return: Two dicts, token_to_index and index_to_token. token_to_index holds the
    """
    token_to_index = {}
    index_to_token = {}
    for name in df[column].values:
        for token in name:
            if token in token_to_index:
                continue
            index = len(token_to_index)
            token_to_index[token] = index
            index_to_token[index] = token
    index = len(token_to_index)
    token_to_index['\n'] = index
    index_to_token[index] = '\n'
    return token_to_index, index_to_token


def create_indices_file(text):
    token_to_index = {}
    index_to_token = {}

    for token in text:
        if token in token_to_index:
            continue
        index = len(token_to_index)
        token_to_index[token] = index
        index_to_token[index] = token

    return token_to_index, index_to_token


def chunk_names_file(text, timesteps, stepsize=1):
    chunks = []
    next_char = []
    for i in range(0, len(text) - timesteps, stepsize):
        chunks.append(text[i:i + timesteps])
        next_char.append(text[i + timesteps])
    return chunks, next_char


def chunk_names(df, col, timesteps):
    chunks = []
    next_char = []
    for row in df.itertuples(index=False):
        text = getattr(row, col)
        for j in range(len(text) - timesteps + 1):
            chunks.append(text[j:j + timesteps])
            if j == len(text) - timesteps:
                next_char.append('\n')
            else:
                next_char.append(text[j + timesteps])
    return chunks, next_char


def create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size):
    assert len(chunks) == len(next_char)
    X = numpy.zeros((len(chunks), timesteps, vocab_size), dtype=numpy.bool)
    y = numpy.zeros((len(chunks), vocab_size), dtype=numpy.bool)
    for i, chunk in enumerate(chunks):
        for t, token in enumerate(chunk):
            index = token_to_index[token]
            X[i, t, index] = 1
        index = token_to_index[next_char[i]]
        y[i, index] = 1
    return X, y


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)


@attr.s
class SampleNames(Callback):
    chunks = attr.ib()
    timesteps = attr.ib()
    vocab_size = attr.ib()
    token_to_index = attr.ib()
    index_to_token = attr.ib()

    def on_epoch_end(self, epoch, logs):
        if epoch % 10 != 0:
            return
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.chunks) - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = self.chunks[start_index]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(50):
                x_pred = numpy.zeros((1, self.timesteps, self.vocab_size))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.token_to_index[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = self.index_to_token[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                if next_char == '\n':
                    break

                sys.stdout.write(next_char)
                sys.stdout.flush()

            print()


@attr.s
class NameGenerator(object):
    timesteps = attr.ib()
    vocab_size = attr.ib()
    token_to_index = attr.ib()
    index_to_token = attr.ib()
    model = attr.ib()

    def generate(self, n=5, seed='^^^', diversity=1.0):
        """
        Uses the trained model to generate new names.
        :param n: the number of names to generate
        :param seed: characters to see the generation. Defaults to ^^^, which are the characters used to signify the start of
        :param diversity: controls the randomness in the generation. Above 1.0 makes the generation more aggressive and
        below 1.0 is more conservative.
        a name.
        :return: a list of generated names
        """
        remove_special = re.compile(rf'[\^$]{{{self.timesteps}}}')
        names = []
        while len(names) < n:
            sequence = seed
            generated = ''
            generated += sequence

            for i in range(50):
                x_pred = numpy.zeros((1, self.timesteps, self.vocab_size), dtype=numpy.bool)
                for t, char in enumerate(sequence):
                    x_pred[0, t, self.token_to_index[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = self.index_to_token[next_index]

                if next_char == '\n':
                    break

                generated += next_char

                # fill up a buffer of size timestep, if its filled drop the first element.
                if len(sequence) == self.timesteps:
                    sequence = sequence[1:] + next_char
                else:
                    sequence += next_char
            generated = remove_special.sub('', generated)
            names.append(generated)
        return names


@attr.s
class SampleNamesFile(Callback):
    timesteps = attr.ib()
    vocab_size = attr.ib()
    token_to_index = attr.ib()
    index_to_token = attr.ib()
    text = attr.ib()

    def on_epoch_end(self, epoch, logs):
        if epoch % 5 != 0:
            return
        # Function invoked at end of each epoch. Prints generated text.
        print(f'\n----- Generating text after Epoch: {epoch}')

        seed = random.choice(string.ascii_lowercase)

        print(f'----- Generating with seed: "{seed}"')

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print(f'----- diversity: {diversity}')

            sequence = seed
            generated = ''
            generated += sequence

            for i in range(50):
                x_pred = numpy.zeros((1, self.timesteps, self.vocab_size))
                for t, char in enumerate(sequence):
                    x_pred[0, t, self.token_to_index[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = self.index_to_token[next_index]

                if next_char == '\n':
                    break

                generated += next_char

                # fill up a buffer of size timestep, if its fill drop the first element.
                if len(sequence) == self.timesteps:
                    sequence = sequence[1:] + next_char
                else:
                    sequence += next_char

            print(generated)
