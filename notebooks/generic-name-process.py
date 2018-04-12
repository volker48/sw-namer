
# coding: utf-8

# In[27]:


import pandas
import numpy
import random
import sys

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, Adam


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


combined = combined.sample(frac=1, random_state=38974)


# In[12]:


combined


# In[13]:


combined.loc[:, 'name'] = combined.name.str.lower()


# In[14]:


combined


# In[15]:


timesteps = 1


# In[16]:


token_to_index, index_to_token = create_indices(combined, 'name')


# In[ ]:


chunks, next_char = chunk_names(combined, 'name', timesteps)


# In[ ]:


vocab_size = len(token_to_index)


# In[ ]:


X, y = create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[17]:


model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, vocab_size)))
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


# In[48]:


timesteps = 2


# In[49]:


chunks, next_char = chunk_names_file(text, timesteps, stepsize=1)


# In[50]:


X, y = create_training_vectors(chunks, next_char, token_to_index, timesteps, vocab_size)


# In[56]:


model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(LSTM(64, input_shape=(timesteps, vocab_size), return_sequences=True))
model.add(LSTM(64, input_shape=(timesteps, vocab_size), return_sequences=False))

model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=.01, clipvalue=1)
model.compile(optimizer, 'categorical_crossentropy')


# In[57]:


model.summary()


# In[ ]:


model.fit(X, y, epochs=200, batch_size=64, callbacks=[SampleNamesFile(timesteps, vocab_size, token_to_index, index_to_token, text)])

