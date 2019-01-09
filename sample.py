#!/usr/bin/ python
# -*- coding: utf-8 -*-

import json
import logging

import numpy as np
import tensorflow as tf

import utils
from model import Model

from flags import parse_args
FLAGS, unparsed = parse_args()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)

dictionary = utils.load_dictionary(FLAGS.dictionary)

reverse_dictionary = utils.load_dictionary(FLAGS.reverse_dictionary)

print(reverse_dictionary[0])
reverse_list = [reverse_dictionary[i]
                for i in range(len(reverse_dictionary))]
titles = ['江神子', '蝶恋花', '渔家傲']


model = Model(learning_rate=FLAGS.learning_rate, batch_size=1, num_steps=1)
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
        exit(0)

    for title in titles:

        # feed title
        for head in title:
            input = utils.index_data(np.array([[head]]), dictionary)
            print(head,input)
            state = sess.run(model.init_state, feed_dict={model.X: input})
            feed_dict = {model.X: input,
                         model.init_state: state,
                         model.keep_prob: 1.0}

            pred, state = sess.run(
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict)

        sentence = title
        #word_index = pred[0].argsort()[-1]
        word_index = np.argmax(pred, axis=-1)
        print("word_index : ",word_index)
        # generate sample
        for i in range(64):

            feed_dict = {model.X: word_index,
                         model.init_state: state,
                         model.keep_prob: 1.0}

            pred, state = sess.run(
                [model.predictions, model.outputs_state_tensor], feed_dict=feed_dict)

            word_index =  np.argmax(pred, axis=-1)
            word = np.take(reverse_list, word_index)
            word = word[0][0][0]
            sentence = sentence + word

        logging.debug('==============[{0}]=============='.format(title))
        logging.debug(sentence)
