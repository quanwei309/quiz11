#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import numpy as np

import tensorflow as tf

import utils
from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


def get_train_data(vocabulary,batch_size,num_steps):
    #集合length
    data_size = len(vocabulary)
    dictionary = utils.load_dictionary(FLAGS.dictionary)
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
    X = []
    Y =[]
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
        X.append(x)
        Y.append(y)
    return zip(X, Y)


vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))

num_steps = FLAGS.num_steps

print('num_steps : ', num_steps)

model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=num_steps)
model.build(embedding_file=FLAGS.embedding_file)




with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')



    for x in range(1):
        logging.debug('epoch [{0}]....'.format(x))
        step = 0
        for X, Y in get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=num_steps):

            if step == 0:
                training_state = sess.run(model.init_state, feed_dict={model.X: X})
            feed_dict = {model.X: X, model.Y: Y,model.init_state:training_state}
            gs, _op, training_state, l, summary_string = sess.run(
                [model.global_step,
                 model.optimizer,
                 model.outputs_state_tensor,
                 model.loss,
                 model.merged_summary_op],
                 feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)
            if gs % 100 == 0:
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
            step +=1
    summary_string_writer.close()



