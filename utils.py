#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np

from flags import parse_args
FLAGS, unparsed = parse_args()

def read_data(filename):
  """
  对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
  """


  # 读取文本，预处理，分词，得到词典
  raw_word_list = []
  with open(filename, "r", encoding='UTF-8') as f:
    line = f.readline()
    while line:
      while '\n' in line:
        line = line.replace('\n', '')
      while ' ' in line:
        line = line.replace(' ', '')
      if len(line) > 0:  # 如果句子非空
        raw_words = list(line)
        raw_word_list.extend(raw_words)
      line = f.readline()
  return raw_word_list


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def load_dictionary(filename):
    inf = np.load(filename)
    dictionary = inf.item()
    return dictionary


def get_train_data(vocabulary,batch_size,num_steps):
    #集合length
    data_size = len(vocabulary)
    dictionary = load_dictionary(FLAGS.dictionary)
    i_data = list()
    unk_count = 0
    #汉字转字典索引编码
    for word in vocabulary:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        i_data.append(index)

    #第N个字符作为x,N=1个字符为Y
    raw_x = [ch for ch in i_data]
    raw_y = [ch for ch in i_data[1:]]
    raw_y.append(len(dictionary) - 1)

    # 得出“分区”数量
    data_partition_size = data_size // batch_size
    data_x = np.zeros([batch_size, data_partition_size], dtype=np.int32)
    data_y = np.zeros([batch_size, data_partition_size], dtype=np.int32)
    print('get_train_data->batch_size : ', batch_size)
    for i in range(batch_size):
        data_x[i] = raw_x[data_partition_size * i:data_partition_size * (i + 1)]
        data_y[i] = raw_y[data_partition_size * i:data_partition_size * (i + 1)]

    epoch_size = data_partition_size // num_steps
    print('get_train_data->epoch_size : ',epoch_size)
    for i in range(epoch_size):

        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
    yield (x, y)










def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
