import tensorflow as tf
import flags
import os
import numpy as np
import pandas as pd
import pickle
import nltk
import json
import codecs
from collections import Counter
from random import randint

FLAGS = tf.app.flags.FLAGS


def get_caption_data(mode="train"):
    if mode == "train":
        feats = np.load("./data/train_feats.npy")
        captions = np.load("./data/train_captions.npy")
        filenames_to_captions = np.load("./data/train_filename_caption_association.npy")
    # elif mode == "val":
    #     feats = np.load("./data/val_feats.npy")
    #     captions = np.load("./data/val_captions.npy")
    #     filenames_to_captions = np.load("./data/val_filename_caption_association.npy")
    else:
        feats = np.load("data/test_feats.npy")
        captions = np.load("data/test_captions.npy")
        filenames_to_captions = np.load("data/test_filename_caption_association.npy")

    return feats, captions, filenames_to_captions


def tokenize(unprocessed_captions):
    processed_captions = []
    for unprocessed_caption in unprocessed_captions:
        processed_caption = [FLAGS.start_word]
        # processed_caption.extend(nltk.tokenize.word_tokenize(unprocessed_caption.lower()))
        processed_caption.extend([w for w in nltk.tokenize.word_tokenize(unprocessed_caption.lower().strip()) if w.isalpha()])
        processed_caption.append(FLAGS.end_word)
        processed_captions.append(processed_caption)

    return processed_captions


def preprocess_for_test(unprocessed_captions):
    processed_captions = []
    for unprocessed_caption in unprocessed_captions:
        processed_caption = (' '.join([w for w in nltk.tokenize.word_tokenize(unprocessed_caption.lower().strip()) if w.isalpha()]))
        processed_captions.append(processed_caption)

    return processed_captions


def preprocess_captions(captions):
    print('pre processing word counts and creating vocab based on word count threshold {}'.format(FLAGS.min_word_count))

    captions = tokenize(captions)

    counter = Counter()
    for caption in captions:
        counter.update(caption)

    print("Nb of words {}".format(len(counter)))

    # take only common words

    common_words = [word for word in counter.items() if word[1] >= FLAGS.min_word_count]

    print("Nb of common words {}".format(len(common_words)))

    # with open(FLAGS.word_counts_output_file, "w") as f:
    #     for word, count in common_words:
    #         f.write("{} {}\n".format(word, count))
    # print("Finished writing word count vocabulary file")

    # create vocab
    # unk_id = len(common_words)

    if not tf.gfile.Exists('data/word_to_index.npy') or not tf.gfile.Exists('data/index_to_word.npy'):
        print("Recreating dictionaries from training captions")
        word_to_index = dict([(word, id) for id, word in enumerate([word_count[0] for word_count in common_words])])
        index_to_word = dict([(id, word) for id, word in enumerate([word_count[0] for word_count in common_words])])
        np.save('data/index_to_word', index_to_word)
        np.save('data/word_to_index', word_to_index)
    else:
        print("Loading dictionaries")
        word_to_index = np.load('data/word_to_index.npy')[()]
        index_to_word = np.load('data/index_to_word.npy')[()]

    maxlen = np.max([len(c) for c in captions])
    # vocab = Vocabulary(vocab_dict, unk_id)

    # word_counts = {}
    # nsents = 0
    # for caption in captions:
    #     nsents += 1
    #     for w in caption.lower().split(' '):
    #         word_counts[w] = word_counts.get(w, 0) + 1
    # vocab = [w for w in word_counts if word_counts[w] >= FLAGS.word_frequency_threshold]
    # print('filtered words from {} to {}'.format(len(word_counts), len(vocab)))
    # index_to_word = {}
    # index_to_word[0] = '.'
    # word_to_index = {}
    # word_to_index['#START#'] = 0
    # index = 1
    # for w in vocab:
    #     word_to_index[w] = index
    #     index_to_word[index] = w
    #     index += 1
    #
    # word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * counter[index_to_word[i]] for i in index_to_word])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return word_to_index, index_to_word, bias_init_vector, maxlen


def load_model(saver, sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))


def get_all_captions_for_filename(filename, filenames_to_captions):
    # filename = filenames_to_captions[index][0]
    reference_captions = [c for f, c in filenames_to_captions if f == filename]
    # reference_captions = [' '.join(str(w) for w in tokenize(c)) for c in reference_captions]
    reference_captions = preprocess_for_test(reference_captions)

    return reference_captions


def compute_bleu_score_for_batch(gen_sent, start, end, filenames_to_captions):
    batch_filenames_for_captions = filenames_to_captions[start:end]
    batch_reference_captions = [get_all_captions_for_filename(f, filenames_to_captions) for f, _ in batch_filenames_for_captions]

    gen_sent = [[w for w in g if w != FLAGS.start_word and w != FLAGS.end_word] for g in gen_sent]
    references_hypothesis_assoc = list(zip(batch_reference_captions, gen_sent))
    cc = nltk.translate.bleu_score.SmoothingFunction()
    bleu_scores = [nltk.translate.bleu_score.sentence_bleu(references, hypothesis, smoothing_function=cc.method3) for references, hypothesis in references_hypothesis_assoc]
    bleu_score_batch = np.mean(bleu_scores)

    return bleu_score_batch


def compute_bleu_score_for_whole_dataset(gen_sent, filenames_to_captions):
    reference_captions = [get_all_captions_for_filename(f, filenames_to_captions) for f, _ in
                          filenames_to_captions]

    gen_sent = [[w for w in g if w != FLAGS.start_word and w != FLAGS.end_word] for g in gen_sent]
    gen_sent = [' '.join(g) for g in gen_sent]
    references_hypothesis_assoc = list(zip(reference_captions, gen_sent))
    cc = nltk.translate.bleu_score.SmoothingFunction()
    bleu_scores = [nltk.translate.bleu_score.sentence_bleu(references, hypothesis, smoothing_function=cc.method3) if len(hypothesis) != 0 else 0 for
                   references, hypothesis in references_hypothesis_assoc]
    bleu_score = np.mean(bleu_scores)

    return bleu_score


def create_eval_json(all_gen_sents, filenames_to_captions):
    all_gen_sents = [[w for w in g if w != FLAGS.start_word and w != FLAGS.end_word] for g in all_gen_sents]
    all_gen_sents = [' '.join(g) for g in all_gen_sents]
    anns = []
    filenames_captions_dict = {}
    for f, c in filenames_to_captions:
        filenames_captions_dict.setdefault(f, [])
        filenames_captions_dict[f].append(c)

    for f in filenames_captions_dict.keys():
        filenames_captions_dict[f] = preprocess_for_test(filenames_captions_dict[f])

    for i in range(len(all_gen_sents)):
        filename, caption = filenames_to_captions[i]
        gen_sent = all_gen_sents[i]
        id_ = list(filenames_captions_dict.keys()).index(filename)
        anns_elem = str(id_) + ',' + str(gen_sent)
        anns.append(anns_elem)

    d = [{'image_id': elem.split(',')[0], "caption": elem.split(',')[1]} for elem in anns]

    json.dump(d, open('./results/flicker_' + FLAGS.validate_on + '_res.json', 'w'))


def get_filename_id(filename, all_filenames_to_captions):
    filenames_captions_dict = {}
    for f, c in all_filenames_to_captions:
        filenames_captions_dict.setdefault(f, [])
        filenames_captions_dict[f].append(c)
    id_ = list(filenames_captions_dict.keys()).index(filename)


def clip_by_value(t_list, clip_value_min, clip_value_max, name=None):
    # if (not isinstance(t_list, collections.Sequence)
    #     or isinstance(t_list, six.string_types)):
    #     raise TypeError("t_list should be a sequence")
    t_list = list(t_list)

    with tf.name_scope(name or "clip_by_value") as name:
        values = [
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i)
            if t is not None else t
            for i, t in enumerate(t_list)]
        values_clipped = []
        for i, v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.get_default_graph().colocate_with(v):
                    values_clipped.append(
                        tf.clip_by_value(v, clip_value_min, clip_value_max))

        list_clipped = [
            tf.IndexedSlices(c_v, t.indices, t.dense_shape)
            if isinstance(t, tf.IndexedSlices)
            else c_v
            for (c_v, t) in zip(values_clipped, t_list)]

    return list_clipped


def decode_sequence(batch_of_seq, index_to_word, current_mask_matrix, maxlen):
    batch_of_sents = [[index_to_word[caption_id] for caption_id in gen_sent if caption_id in index_to_word] for gen_sent
                      in batch_of_seq]
    caption_sizes = np.sum(current_mask_matrix, axis=1).tolist()
    batch_of_sents = [gen_sent[:min(int(s), maxlen)] for gen_sent, s in zip(batch_of_sents, caption_sizes)]
    return batch_of_sents


def get_filename_image_id_associations(mode):
    print("Loading filename image id associations")
    image_id_to_filename = np.load('data/' + mode + '_image_id_to_filename.npy')[()]
    filename_to_image_id = np.load('data/' + mode + '_filename_to_image_id.npy')[()]

    return image_id_to_filename, filename_to_image_id