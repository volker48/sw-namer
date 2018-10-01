
# coding: utf-8

# In[ ]:


import pandas
import numpy
import random
import sys

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, GRU, Dropout
from keras.optimizers import RMSprop, Adam, SGD


# In[ ]:


cd ..


# In[ ]:


from swnamer.process import *


# In[ ]:


legend_names = pandas.read_csv('data/legend_names.csv')


# In[ ]:


canon_names = pandas.read_csv('data/cannon_names.csv')


# In[ ]:


cast_names = pandas.read_csv('data/cast_names.csv')


# In[ ]:


clone_wars_names = pandas.read_csv('data/clone_wars.csv')


# In[ ]:


kotor_names = pandas.read_csv('data/kotor.csv')


# In[ ]:


combined = pandas.concat((legend_names, canon_names, cast_names, clone_wars_names, kotor_names))


# In[ ]:


combined = combined.reset_index(drop=True)


# In[ ]:


combined.shape


# In[ ]:


combined.drop_duplicates().shape


# In[ ]:


combined = combined.drop_duplicates()


# In[ ]:


combined.sample(10)


# In[ ]:


combined.loc[:, 'name'] = combined.name.str.lower()


# In[ ]:


timesteps = 4


# Since the RNN will be predicting the probability of a character based on the last timestep chars in the sequence
# we pad each name with a special character.

# In[ ]:


combined['name'] = ('^' * timesteps) + combined.name + ('$' * timesteps)


# In[ ]:


combined.loc[:, 'length'] = combined.name.str.len()


# In[ ]:


combined.sample(10)


# In[ ]:


combined.length.max()


# In[ ]:


combined.to_csv('output/starwars_processed.csv', index=False)


# In[ ]:


token_to_index, index_to_token = create_indices(combined, 'name')


# In[ ]:


chars = token_to_index.keys()


# In[ ]:


chars


# In[ ]:


vocab_size = len(chars)
vocab_size


# In[ ]:


chunks, next_char = chunk_names(combined, 'name', timesteps)


# In[ ]:


X, y = create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size)


# In[ ]:


model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, vocab_size), return_sequences=True,))
model.add(LSTM(32, input_shape=(timesteps, vocab_size), return_sequences=False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = SGD(lr=.01, momentum=.99, nesterov=True)
model.compile(optimizer, 'categorical_crossentropy')
model.summary()


# In[ ]:


sampler = SampleNames(chunks, timesteps, vocab_size, token_to_index, index_to_token)
earlystopping = EarlyStopping(mode='min', patience=7, min_delta=.001)
checkpoint = ModelCheckpoint('output/starwars.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
callbacks = [sampler, earlystopping, checkpoint]


# In[ ]:


model.fit(X, y, epochs=100, batch_size=32, validation_split=.1, callbacks=callbacks)


# In[ ]:


#model.save('output/starwars-2-lstm-64.hdf5')


# In[ ]:


model = load_model('output/starwars.23-1.94.hdf5')


# In[ ]:


gen = NameGenerator(timesteps, vocab_size, token_to_index, index_to_token, model)


# In[ ]:


gen.generate(seed='^^^^', diversity=1.)


# In[ ]:


gen.generate(10, seed='dar')


# In[ ]:


gen.generate(10)


# In[ ]:


#commander papano sasnaphared borbilz


# In[ ]:


combined[combined.name.str.contains('fn')]


# In[ ]:


combined.sample(10)

