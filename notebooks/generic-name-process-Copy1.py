
# coding: utf-8

# In[1]:


import pandas
import numpy
import random
import sys

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, Adam, SGD


# In[2]:


cd ..


# In[3]:


from swnamer.process import chunk_names, chunk_names_file, create_indices_file, create_indices, SampleNames, SampleNamesFile, create_training_vectors


# In[4]:


male = pandas.read_csv('data/male.txt', header=5, names=['name'])


# In[5]:


female = pandas.read_csv('data/female.txt', header=5, names=['name'])


# In[6]:


male.shape, female.shape


# In[7]:


male.columns


# In[8]:


male.sample(20)


# In[9]:


female.sample(20)


# In[10]:


combined = pandas.concat([male, female])


# In[11]:


# shuffle
combined = combined.sample(frac=1, random_state=38974)


# In[12]:


combined


# In[13]:


combined.loc[:, 'name'] = combined.name.str.lower()


# In[14]:


combined


# In[16]:


timesteps = 3


# In[17]:


combined['name'] = ('^' * timesteps) + combined.name


# In[18]:


token_to_index, index_to_token = create_indices(combined, 'name')


# In[19]:


chunks, next_char = chunk_names(combined, 'name', timesteps)


# In[21]:


vocab_size = len(token_to_index)
vocab_size


# In[ ]:


X, y = create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, vocab_size)))
model.add(LSTM(128, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = Adam(lr=.01, clipvalue=5)
model.compile(optimizer, 'categorical_crossentropy')


# In[ ]:


model.fit(X, y, epochs=200, batch_size=128, callbacks=[SampleNames(chunks, timesteps, vocab_size, token_to_index, index_to_token)])


# In[ ]:


combined.to_csv('output/standard_names.csv', index=False, header=False)


# In[4]:


with open('output/standard_names.csv', 'r') as infile:
    text = infile.read()


# In[5]:


token_to_index, index_to_token = create_indices_file(text)


# In[6]:


vocab_size = len(token_to_index)


# In[7]:


timesteps = 2


# In[8]:


chunks, next_char = chunk_names_file(text, timesteps, stepsize=1)


# In[9]:


X, y = create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size)


# In[10]:


train_end = int(X.shape[0] * .8)


# In[11]:


X_train, X_valid, y_train, y_valid = X[:train_end], X[train_end:], y[:train_end], y[train_end:]


# In[12]:


model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, vocab_size), return_sequences=False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = SGD(lr=.01, momentum=.99, nesterov=True, clipvalue=1)
model.compile(optimizer, 'categorical_crossentropy')


# In[13]:


model.summary()


# In[14]:


model.fit(X_train, y_train, epochs=81, batch_size=128, 
          callbacks=[SampleNamesFile(timesteps, vocab_size, token_to_index, index_to_token, text)],
         validation_data=(X_valid, y_valid))


# In[15]:


model.save('output/generic-names-81-epochs.hdf5')


# In[16]:


token_to_index.keys()


# In[ ]:


com

