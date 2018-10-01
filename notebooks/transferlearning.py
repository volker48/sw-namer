
# coding: utf-8

# In[23]:


import pandas
import numpy
import random
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, GRU, Dropout
from keras.optimizers import RMSprop, Adam, SGD

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cd ..


# In[3]:


from swnamer.process import *


# In[4]:


names = pandas.read_csv('output/names.csv')


# In[5]:


names.sample(5)


# In[6]:


starwars_names = pandas.read_csv('output/starwars_processed.csv')


# In[7]:


starwars_names.sample(5)


# In[8]:


timesteps = 3


# In[9]:


combined = pandas.concat((names, starwars_names), axis=0)


# In[10]:


token_to_index, index_to_token = create_indices(combined, 'name')


# In[11]:


chars = token_to_index.keys()


# In[12]:


chars


# In[13]:


vocab_size = len(chars)
vocab_size


# In[14]:


chunks, next_char = chunk_names(names, 'name', timesteps)


# In[15]:


X, y = create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size)


# In[28]:


model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(LSTM(64, input_shape=(timesteps, vocab_size), return_sequences=False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = SGD(lr=.01, momentum=.99, nesterov=True)
model.compile(optimizer, 'categorical_crossentropy')
model.summary()


# In[29]:


sampler = SampleNames(chunks, timesteps, vocab_size, token_to_index, index_to_token)
earlystopping = EarlyStopping(mode='min', patience=7, min_delta=.001)
checkpoint = ModelCheckpoint('output/names.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
callbacks = [sampler, earlystopping, checkpoint]


# In[30]:


history = model.fit(X, y, epochs=100, batch_size=32, validation_split=.2, callbacks=callbacks)


# In[31]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('categorical cross entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[33]:


model = Sequential()
model.add(LSTM(80, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(Dropout(.25))
model.add(LSTM(80, input_shape=(timesteps, vocab_size), return_sequences=False))
model.add(Dropout(.25))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = SGD(lr=.01, momentum=.99, nesterov=True, clipvalue=1)
model.compile(optimizer, 'categorical_crossentropy')
model.summary()


# In[34]:


sampler = SampleNames(chunks, timesteps, vocab_size, token_to_index, index_to_token)
earlystopping = EarlyStopping(mode='min', patience=7, min_delta=.001)
checkpoint = ModelCheckpoint('output/names.{epoch:02d}-{val_loss:.2f}.hdf5')
callbacks = [sampler, earlystopping, checkpoint]


# In[35]:


history = model.fit(X, y, epochs=100, batch_size=64, validation_split=.2, callbacks=callbacks)


# In[36]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('categorical cross entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[63]:


min_index = numpy.argmin(history.history['val_loss'])
min_index, history.history['val_loss'][min_index]


# In[64]:


model = load_model('output/names.40-1.42.hdf5')


# In[65]:


model.layers


# In[66]:


gen = NameGenerator(timesteps, vocab_size, token_to_index, index_to_token, model)


# In[72]:


gen.generate(seed='^^^')


# ## Freeze first two layers

# In[73]:


for layer in model.layers[:2]:
    layer.trainable = False


# In[74]:


optimizer = SGD(lr=.01, momentum=.99, nesterov=True)
model.compile(optimizer, 'categorical_crossentropy')
model.summary()


# In[75]:


starwars_chunks, starwars_next_char = chunk_names(starwars_names, 'name', timesteps)


# In[76]:


X_sw, y_sw = create_training_vectors(starwars_chunks, starwars_next_char, token_to_index, timesteps, vocab_size)


# In[77]:


sampler = SampleNames(starwars_chunks, timesteps, vocab_size, token_to_index, index_to_token)
earlystopping = EarlyStopping(mode='min', patience=10, min_delta=.001)
checkpoint = ModelCheckpoint('output/starwars-transfer.{epoch:02d}-{val_loss:.2f}.hdf5')
callbacks = [sampler, earlystopping, checkpoint]


# In[78]:


history = model.fit(X_sw, y_sw, epochs=100, batch_size=32, validation_split=.1, callbacks=callbacks)


# In[79]:


history = _78


# In[80]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('categorical cross entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# ### Looks like overfitting

# In[88]:


model = load_model('output/names.40-1.42.hdf5')


# In[89]:


# Freeze LSTM layers
model.layers[0].trainable = False
model.layers[2].trainable = False


# In[90]:


optimizer = SGD(lr=.01, momentum=.99, nesterov=True, clipvalue=1)
model.compile(optimizer, 'categorical_crossentropy')


# In[91]:


sampler = SampleNames(starwars_chunks, timesteps, vocab_size, token_to_index, index_to_token)
earlystopping = EarlyStopping(mode='min', patience=10, min_delta=.001)
checkpoint = ModelCheckpoint('output/starwars-transfer.{epoch:02d}-{val_loss:.2f}.hdf5')
callbacks = [sampler, earlystopping, checkpoint]


# In[92]:


history = model.fit(X_sw, y_sw, epochs=100, batch_size=32, validation_split=.1, callbacks=callbacks)


# In[93]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('categorical cross entropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[98]:


numpy.argmin(history.history['val_loss'])


# In[99]:


model = load_model('output/starwars-transfer.10-2.05.hdf5')


# In[100]:


gen = NameGenerator(timesteps, vocab_size, token_to_index, index_to_token, model)


# In[108]:


gen.generate(n=50, diversity=.85)


# In[ ]:


model.save()

