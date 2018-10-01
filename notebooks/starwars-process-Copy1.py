
# coding: utf-8

# In[165]:


import pandas
import numpy
import random
import sys

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, Adam


# In[163]:


cd ..


# In[164]:


from swnamer.process import chunk_names, create_indices


# In[130]:


legend_names = pandas.read_csv('../data/legend_names.csv')


# In[131]:


canon_names = pandas.read_csv('../data/cannon_names.csv')


# In[132]:


cast_names = pandas.read_csv('../data/cast_names.csv')


# In[133]:


clone_wars_names = pandas.read_csv('../data/clone_wars.csv')


# In[134]:


kotor_names = pandas.read_csv('../data/kotor.csv')


# In[135]:


combined = pandas.concat((legend_names, canon_names, cast_names, clone_wars_names, kotor_names))


# In[136]:


combined = combined.reset_index(drop=True)


# In[137]:


combined.shape


# In[138]:


combined.drop_duplicates().shape


# In[139]:


combined = combined.drop_duplicates()


# In[140]:


combined


# In[141]:


combined.loc[:, 'name'] = combined.name.str.lower()


# In[142]:


combined.loc[:, 'length'] = combined.name.str.len()


# In[143]:


combined.length.max()


# In[144]:


combined[combined.length == 27]


# In[145]:


combined.to_csv('../output/starwars_processed.csv', index=False)


# In[166]:


token_to_index, index_to_token = create_indices(combined, 'name')


# In[167]:


chars = token_to_index.keys()


# In[168]:


vocab_size = len(chars)
vocab_size


# In[178]:


timesteps = 1


# In[179]:


chunks, next_char = chunk_names(combined, 'name', timesteps)


# In[180]:


X = numpy.zeros((len(chunks), timesteps, vocab_size))
y = numpy.zeros((len(chunks), vocab_size))


# In[181]:


for i, chunk in enumerate(chunks):
    for t, token in enumerate(chunk):
        index = token_to_index[token]
        X[i, t, index] = 1
    index = token_to_index[next_char[i]]
    y[i, index] = 1


# In[183]:


model = Sequential()
model.add(GRU(128, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(GRU(128))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=.1, clipvalue=12)
model.compile(optimizer, 'categorical_crossentropy')


# In[184]:


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)


def on_epoch_end(epoch, logs):
    if epoch % 10 != 0:
        return
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(chunks) - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = chunks[start_index]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(50):
            x_pred = numpy.zeros((1, timesteps, vocab_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, token_to_index[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = index_to_token[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            
            if next_char == '\n':
                break

            sys.stdout.write(next_char)
            sys.stdout.flush()
    
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


# In[185]:


model.fit(X, y, epochs=200, batch_size=128, callbacks=[print_callback])

