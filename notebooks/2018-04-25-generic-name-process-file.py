
# coding: utf-8

# In[1]:


import pandas
import numpy
import random
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, Adam, SGD

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cd ..


# In[60]:


from swnamer.process import *


# In[5]:


names = pandas.read_csv('output/names.csv')
sw_names = pandas.read_csv('output/starwars_processed.csv')


# In[6]:


combined = pandas.concat([names, sw_names])


# In[10]:


timesteps = 3


# In[9]:


token_to_index, index_to_token = create_indices(combined, 'name')


# In[11]:


with open('output/names.csv') as infile:
    infile.readline()
    names_text = infile.read()


# In[15]:


names_text = names_text.lower()


# In[17]:


chunks, next_char = chunk_names_file(names_text, timesteps)


# In[18]:


vocab_size = len(token_to_index)
vocab_size


# In[19]:


X, y = create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size)


# In[20]:


X.shape


# In[21]:


y.shape


# In[40]:


model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(LSTM(64, input_shape=(timesteps, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = Adam(lr=.01, clipvalue=5)
model.compile(optimizer, 'categorical_crossentropy')


# In[41]:


es = EarlyStopping(mode='min', patience=7, min_delta=.001)
sampler = SampleNamesFile(timesteps, vocab_size, token_to_index, index_to_token, names_text)
checkpoint = ModelCheckpoint('output/names-base-model.hdf5', save_best_only=True)
callbacks = [sampler, es, checkpoint]


# In[42]:


history = model.fit(X, y, validation_split=.2, epochs=200, batch_size=32, callbacks=callbacks)


# In[62]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('categorical cross entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[43]:


gen = NameGenerator(timesteps, vocab_size, token_to_index, index_to_token, model)


# In[59]:


gen.generate(10, seed='')


# In[63]:


model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(Dropout(.5))
model.add(LSTM(64, input_shape=(timesteps, vocab_size)))
model.add(Dropout(.5))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = Adam(lr=.01, clipvalue=5)
model.compile(optimizer, 'categorical_crossentropy')

es = EarlyStopping(mode='min', patience=7, min_delta=.001)
sampler = SampleNamesFile(timesteps, vocab_size, token_to_index, index_to_token, names_text)
checkpoint = ModelCheckpoint('output/names-base-model.hdf5', save_best_only=True)
callbacks = [sampler, es, checkpoint]

history = model.fit(X, y, validation_split=.2, epochs=200, batch_size=32, callbacks=callbacks)


# In[64]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('categorical cross entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[65]:


gen = NameGenerator(timesteps, vocab_size, token_to_index, index_to_token, model)


# In[75]:


gen.generate(10, seed='dus', diversity=1.5)


# In[74]:


model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = Adam(lr=.01, clipvalue=5)
model.compile(optimizer, 'categorical_crossentropy')

es = EarlyStopping(mode='min', patience=7, min_delta=.001)
sampler = SampleNamesFile(timesteps, vocab_size, token_to_index, index_to_token, names_text)
checkpoint = ModelCheckpoint('output/names-base-model.hdf5', save_best_only=True)
callbacks = [sampler, es, checkpoint]

history = model.fit(X, y, validation_split=.2, epochs=200, batch_size=32, callbacks=callbacks)


# In[76]:


gen = NameGenerator(timesteps, vocab_size, token_to_index, index_to_token, model)


# In[80]:


gen.generate(10, seed='', diversity=1)

