import tensorflow as tf
import flags
import os
import numpy as np
import pandas as pd
import pickle
import nltk
import json
import codecs
from random import randint
FLAGS = tf.app.flags.FLAGS


def get_caption_data(mode="train"):
    if mode == "train":
        feats = np.load("./data/train_feats.npy")
        captions = np.load("./data/train_captions.npy")
        filenames_to_captions = np.load("./data/train_filename_caption_association.npy")
    elif mode == "val":
        feats = np.load("./data/val_feats.npy")
        captions = np.load("./data/val_captions.npy")
        filenames_to_captions = np.load("./data/val_filename_caption_association.npy")
    else:
        feats = np.load("data/test_feats.npy")
        captions = np.load("data/test_captions.npy")
        filenames_to_captions = np.load("data/test_filename_caption_association.npy")

    return feats, captions, filenames_to_captions


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


def get_all_captions_for_filename(filename, filenames_to_captions):
    # filename = filenames_to_captions[index][0]
    reference_captions = [c for f, c in filenames_to_captions if f == filename]

    return reference_captions


def compute_bleu_score_for_batch(gen_sent, start, end, filenames_to_captions):
    batch_filenames_for_captions = filenames_to_captions[start:end]
    batch_reference_captions = [get_all_captions_for_filename(f, filenames_to_captions) for f, _ in batch_filenames_for_captions]

    references_hypothesis_assoc = list(zip(batch_reference_captions, gen_sent))
    cc = nltk.translate.bleu_score.SmoothingFunction()
    bleu_scores = [nltk.translate.bleu_score.sentence_bleu(references, hypothesis, smoothing_function=cc.method3) for references, hypothesis in references_hypothesis_assoc]
    bleu_score_batch = np.mean(bleu_scores)

    return bleu_score_batch


def create_eval_json(mode):
    feats, captions, filenames_to_captions = get_caption_data(mode=mode)

    # lets give fixed values to mandatory fields
    license_ = 3
    url_ = 'asdasdsda.com'
    width_ = 640
    height_ = 480
    date_captured = 14

    out_json_tr = []
    captions_tr = []
    ims = []
    anns = []
    # captions_en = []
    offset = 0
    found = 0
    id_ = 0
    index_ = 0

    filenames_captions_dict = {}
    for f, c in filenames_to_captions:
        filenames_captions_dict.setdefault(f, [])
        filenames_captions_dict[f].append(c)

    for j, f in enumerate(filenames_captions_dict.keys()):

        ims_elem = str(license_) + ',' + str(url_) + ',' + str(f) + ',' + str(j) + ',' + str(
            width_) + ',' + str(date_captured) + ',' + str(height_)
        ims.append(ims_elem)
        for k in range(5):
            anns_elem = str(j) + ',' + str(randint(4000, 9000)) + ',' + str(filenames_captions_dict[f][k])
            anns.append(anns_elem)

    d = {"images": [{'license': elem.split(',')[0], "url": elem.split(',')[1], "file_name": elem.split(',')[2],
                     "id": str(elem.split(',')[3]), "width": elem.split(',')[4], "date_captured": elem.split(',')[5],
                     "height": elem.split(',')[6]} for elem in ims],
         "annotations": [{'image_id': str(elem.split(',')[0]), "id": elem.split(',')[1], "caption": elem.split(',')[2]}
                         for
                         elem in anns]}

    # actually it is the test baseline
    json.dump(d, open('./results/flickr30k_val_baseline.json', 'w'))
