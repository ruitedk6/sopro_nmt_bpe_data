
# coding: utf-8

# In[5]:

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import re
import random
#import utils
import os
sess = tf.InteractiveSession()


# In[6]:

def read_in_files(textfile_eng, textfile_ara, max_len):
    """Reads in english and arabic training files, thows out sentences which are longer than the max_len and
    returns a list of lists ([this, is, the, first, sentence], [this, the, second], ...) for each language"""
    with open(textfile_eng, encoding='utf8') as eng:
        with open(textfile_ara, encoding='utf8') as ara:
            text_eng = eng.read()
            text_ara = ara.read()
            sents_eng = text_eng.split('\n')
            sents_ara = text_ara.split('\n')
            words_eng = list()
            words_ara = list()
            for index in range(len(sents_eng)):
                if index % 100000 == 0:
                    print('Sentence number ' + str(index))
                if len(sents_eng[index].split(' ')) <= max_len:
                    words_eng.append(sents_eng[index].split(' '))
                    words_ara.append(sents_ara[index].split(' '))   
    
    return words_eng, words_ara


# In[7]:

def get_joint_vocab(words_eng, words_ara):
    """takes lists of lists as input, writes a joint vocabulary file and creates a word2index and index2word dictionary.
    ex. word2index = {dog: 23, cat: 393, ...}
    ex. index2word = {23: dog, 393: cat, ...}"""
    word2index = dict()
    index2word = dict()
    special_chars = ['PAD', 'EOS']
    with open('vocab.joint.tsv', 'w+') as f:
        index = 0
        
        for char in special_chars:
            word2index[char] = index
            index2word[index] = char
            f.write(char + '\n')
            index +=1
        
        for sent in words_eng:
            for word in sent:
                if word not in word2index:
                    word2index[word] = index
                    index2word[index] = word
                    f.write(word + '\n')
                    index += 1
        
        for sent in words_ara:
            for word in sent:
                if word not in word2index:
                    word2index[word] = index
                    index2word[index] = word
                    f.write(word + '\n')
                    index += 1
                
    return word2index, index2word


# In[8]:

def get_indexed_words(words, word2index):
    """takes a list of lists (with words) and a word2index dict as input, and returns a list of lists (with indexes)
    ex. indexed_words = [[393, 32, 1, 34923, 3], [2830, 2, 3, 435, 9304, 393], ...]"""
    indexed_words = list()
    
    for sent in words:
        cur_sent = list()
        
        for word in sent:
            cur_sent.append(word2index[word])
            
        indexed_words.append(cur_sent)
        
    return indexed_words
            


# In[9]:

def order_by_size(indexed_eng, indexed_ara):
    """takes the list of lists (with indexes) for english and arabic as input, and orders the sentences according to
    the length of the english sentence.
    returns a sorted version of indexed_eng and its corresponding counterpart in arabic"""
    indexed_sents = dict()
    for index in range(len(indexed_eng)):
        indexed_sents[str(indexed_eng[index])] = index
    
    indexed_eng.sort(key=len)
    sorted_eng = indexed_eng
    
    sorted_ara = list()
    for sent in sorted_eng:
        sorted_ara.append(indexed_ara[indexed_sents[str(sent)]])
        
    return sorted_eng, sorted_ara
    


# In[10]:

def generate_sample(indexed_eng, indexed_ara):
    """takes the sorted lists of lists as input and returns an iterator object over these
    this iterator object yields a source (eng) and its corresponding target (ara) sentence whenever next is called"""
    #this is what enumerate(indexed_eng) looks like -> [(0, [033, 1283, 393]), (1, [...]), ...]
    for i, source in enumerate(indexed_eng):
        target = indexed_ara[i]
        yield source, target


# In[11]:

def get_batch(iterator, batch_size):
    """takes the generate_sample iterator and a batch_size as input and returns a generator object that yields
    corresponding eng and ara batches
    ex. source_batch = [[3, 28342, 239, 79, 273, 570], [383, 8234832, 19293, 19, 394], ...]"""
    while True:
        #put batch in a list
        batch = list()
        for index in range(batch_size):
            batch.append(next(iterator))
                
        #get max_len for this batch for eng and ara
        max_len_eng = 0
        max_len_ara = 0
        for sent in batch:
            if len(sent[0]) > max_len_eng:
                max_len_eng = len(sent[0])
            if len(sent[1]) > max_len_ara:
                max_len_ara = len(sent[1])
        
        #add padding to sentences shorter than corresponding max_len
        for sent in range(len(batch)):
            dif = max_len_eng - len(batch[sent][0])
            if dif != 0:
                for i in range(dif):
                    batch[sent][0].append(0)
            dif = max_len_ara - len(batch[sent][1])
            if dif != 0:
                for i in range(dif):
                    batch[sent][1].append(0)
            
                
        #create empty matrices of size [batch_size, max_len] to fill with current batch        
        source_batch = np.zeros((batch_size, max_len_eng), dtype=np.int32)
        target_batch = np.zeros((batch_size, max_len_ara), dtype=np.int32)
        #fill with batch
        for index in range(batch_size):
            source_batch[index], target_batch[index] = batch[index]
            
        yield source_batch, target_batch
        


# In[40]:

def prepare_data(text_eng, text_ara, max_len, batch_size):
    print('Reading in training files...')
    words_eng, words_ara = read_in_files(text_eng, text_ara, max_len)
    print('Creating vocabulary file...')
    word2index, index2word = get_joint_vocab(words_eng, words_ara)
    print('Indexing words...')
    indexed_eng = get_indexed_words(words_eng, word2index)
    indexed_ara = get_indexed_words(words_ara, word2index)
    print('Prepare for bucketing...')
    sorted_eng, sorted_ara = order_by_size(indexed_eng, indexed_ara)
    print('Create iterator...')
    iterator = generate_sample(sorted_eng, sorted_ara)
    print('Generate batches...')
    
    
    return get_batch(iterator, batch_size)


# In[41]:

#Note: The way the data is structured now [batch_size, max_len], time_major must be set to false!
#Returns an iterator object which yields a source and target batch of size [batch_size, max_len]
#where max_len varies between buckets
#batch_gen = prepare_data('test1.bpe.eng', 'test.bpe.ara', 50, 128)
#print(list(batch_gen)[:128])


