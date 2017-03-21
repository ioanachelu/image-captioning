import tensorflow as tf
import flags
import os
import numpy as np
import pandas as pd
import pickle
FLAGS = tf.app.flags.FLAGS


def get_caption_data(mode="train"):
    if mode == "train":
        feats = np.load("./data/train_feats.npy")
        captions = np.load("./data/train_captions.npy")
    elif mode == "val":
        feats = np.load("./data/val_feats.npy")
        captions = np.load("./data/val_captions.npy")
    else:
        feats = np.load("./data/test_feats.npy")
        captions = np.load("./data/test_captions.npy")

    return feats, captions


def preprocess_captions(captions):
    print('pre processing word counts and creating vocab based on word count threshold {}'.format(FLAGS.word_frequency_threshold))
    word_counts = {}
    nsents = 0
    for caption in captions:
        nsents += 1
        for w in caption.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= FLAGS.word_frequency_threshold]
    print('filtered words from {} to {}'.format(len(word_counts), len(vocab)))
    index_to_word = {}
    index_to_word[0] = '.'
    word_to_index = {}
    word_to_index['#START#'] = 0
    index = 1
    for w in vocab:
        word_to_index[w] = index
        index_to_word[index] = w
        index += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[index_to_word[i]] for i in index_to_word])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return word_to_index, index_to_word, bias_init_vector


def load_model(saver, sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))
