#!/usr/bin/env python
# coding: utf-8

# 
# As of keras 2, the module keras.layers.merge doesn't have a generic public Merge-Layer. Instead you are supposed to import the subclasses like keras.layers.Add or keras.layers.Concatenate etc. directly (or their functional interfaces with the same names lowercase: keras.layers.add, keras.layers.concatenate etc.).

# In[1]:


#!pip install np_utils
# the above is more explicit about where the package will be installed.. but this is on aws...should I worry?

#import sys
#!{sys.executable} -m pip install np_utils


# In[1]:


#!{sys.executable} -m pip install keras


# Yes, there are a lot of libraries to be installed.
# 

# In[3]:


import numpy as np
import pandas as pd
#import cPickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys 
import os

os.environ['KERAS_BACKEND']='theano'

#import keras
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils.np_utils import to_categorical

#from keras.layers import Embedding
#from keras.layers import Dense, Input, Flatten
#from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
#from keras.models import Model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#import np_utils is the same as utils apparently...This is the conclusion I've come to
from tensorflow.keras import utils

# for embedding
from tensorflow.keras.layers import Embedding
# what for i forgot
from tensorflow.keras.layers import Dense, Input, Flatten
# convolution. Merge is already builtin keras 2. It is not a -layer anymore. Use 'keras.[function]' directly.
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model


# In[4]:


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# ### Beautiful Soup
# Use Beautiful Soup to remove HTML tags and unwanted characters.

# In[5]:


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')
print('Shape of dataset: ', data_train.shape)

texts = []
labels = []


# In[7]:


for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx])
    #texts.append(clean_str(str(text.get_text().encode('ascii','ignore'))))
    texts.append(clean_str(str(text.get_text().encode('ascii', 'ignore'))))
    labels.append(data_train.sentiment[idx])


# In[6]:


macronum=sorted(set(data_train['sentiment']))
macro_to_id = dict((note, number) for number, note in enumerate(macronum))

def fun(i):
    return macro_to_id[i]

data_train['sentiment']=data_train['sentiment'].apply(fun)


# ### We need the Keras parts from now on. We start by tokenizing our data.
# 
# - Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
# - Words that have fewer than 3 characters are removed.
# - All stopwords are removed.
# - Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
# - Words are stemmed — words are reduced to their root form.
# 
# "UserWarning: The `nb_words` argument in `Tokenizer` has been renamed `num_words`.
#   warnings.warn('The `nb_words` argument in `Tokenizer`"

# In[8]:


import tensorflow as tf


# In[9]:


#import keras
from tensorflow.keras.preprocessing.text import Tokenizer


# In[2]:


# TOKENIZEEEE
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# Learning about padding:
# 
# What happens when you apply three 5 x 5 x 3 filters to a 32 x 32 x 3 input volume? The output volume would be 28 x 28 x 3. Notice that the spatial dimensions decrease. As we keep applying conv layers, the size of the volume will decrease faster than we would like. In the early layers of our network, we want to preserve as much information about the original input volume so that we can extract those low level features. Let’s say we want to apply the same conv layer but we want the output volume to remain 32 x 32 x 3. To do this, we can apply a zero padding of size 2 to that layer. Zero padding pads the input volume with zeros around the border. If we think about a zero padding of two, then this would result in a 36 x 36 x 3 input volume. 

# In[11]:


# PADDINGGGG
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = utils.to_categorical(np.asarray(labels))
print('Shape of Data Tensor:', data.shape)
print('Shape of Label Tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


# In[12]:


print('Number of positive and negative reviews in training and validation set ')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))


# #### EMBEDDING
# 
# " GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. " - from GloVe webite
# For an unknown word, the following code will just randomise its vector. 
# 
# Try to specify file path when using open() in jupyter

# In[13]:


glove_dir = "/"
embeddings_index = {}
f = open('glove.6B.100d.txt')
# the below doesn't work in Juptyer...
#f = open('glove.6B.100d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))


# np.to_categorical = numpy version of utils.to_categorical
# Seems like they are doing the same thing though. So I didn't do anymore steps.

# In[14]:


y = [1, 2, 3, 5]
y1 = utils.to_categorical(y)
print(y1)


# In[ ]:





# In[15]:


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)


# ### CNN part
#  
# I have used a 1D Convolutional Neural Network. It uses 128 filters with size 5 and max pooling of 5 and 35. I followed documentation from the Keras website:
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# 
# I printed out the outputs to make sure.

# In[3]:


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
print(sequence_input.get_shape())

embedded_sequences = embedding_layer(sequence_input)
print(embedded_sequences.get_shape())

l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
print(l_cov1.get_shape())

l_pool1 = MaxPooling1D(5)(l_cov1)
print(l_pool1.get_shape())

l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
print(l_cov1.get_shape())

l_pool2 = MaxPooling1D(5)(l_cov2)
print(l_pool2.get_shape())

l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
print(l_cov3.get_shape())

l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
print(l_pool3.get_shape())

l_flat = Flatten()(l_pool3)
print(l_flat.get_shape())

l_dense = Dense(128, activation='relu')(l_flat)
print(l_dense.get_shape())

preds = Dense(len(macronum), activation='softmax')(l_dense)
print(preds.get_shape())


# #### Here we attempt to train the neural network, but the kernel dies every time we use the keras.fit() to train it.
# It says that it will take too much memory. There are similar cases on google, and people advised to not use Sagemaker to run it and use our personal computers instead.

# In[17]:


model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("simplified convolutional neural network")
model.summary()
#model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)


# In[ ]:


history=model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=1, verbose = 0, batch_size=128)


# In[ ]:





# In[ ]:




