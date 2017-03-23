import tensorflow as tf
import flags
import os
from network import ShowAndTell
import os
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import time
from utils import get_caption_data, preprocess_captions, load_model, compute_bleu_score_for_batch, tokenize, compute_bleu_score_for_whole_dataset

FLAGS = tf.app.flags.FLAGS


def run():
    feats, captions, filenames_to_captions = get_caption_data(mode=FLAGS.validate_on)
    word_to_index, index_to_word, bias_init_vector, maxlen = preprocess_captions(captions)
    num_examples_per_epoch = len(range(0, len(feats), FLAGS.batch_size))

    sess = tf.InteractiveSession()
    n_words = len(word_to_index)

    network = ShowAndTell(
        image_embedding_size=FLAGS.image_embedding_size,
        num_lstm_units=FLAGS.num_lstm_units,
        embedding_size=FLAGS.embedding_size,
        batch_size=FLAGS.batch_size,
        n_lstm_steps=maxlen,
        n_words=n_words,
        bias_init_vector=bias_init_vector)

    loss, image, sentence, mask, generated_sentence = network.build_model()
    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    train_op = network.train(global_step, num_examples_per_epoch)
    summary_writer, summaries = network.summary()

    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    load_model(loader, sess)
    sess.run(tf.local_variables_initializer())

    increment_global_step = global_step.assign_add(1)

    all_gen_sents = []
    for start, end in zip(range(0, len(feats), FLAGS.batch_size),
                          range(FLAGS.batch_size, len(feats), FLAGS.batch_size)):
        current_feats = feats[start:end]
        current_captions = captions[start:end]
        current_captions = tokenize(current_captions)
        current_caption_ind = [[word_to_index[word] for word in cap if word in word_to_index] for cap in
                               current_captions]

        current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen)

        current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
        nonzeros = np.array([(x != 0).sum() for x in current_caption_matrix])

        for ind, row in enumerate(current_mask_matrix):
            row[:nonzeros[ind]] = 1

        gen_sent_batch = sess.run(
            generated_sentence, feed_dict={
                image: current_feats,
                sentence: current_caption_matrix,
                mask: current_mask_matrix,
            })
        gen_sent_batch = [[index_to_word[caption_id] for caption_id in gen_sent if caption_id in index_to_word] for
                          gen_sent in list(gen_sent_batch)]
        caption_sizes = np.sum(current_mask_matrix, axis=1).tolist()
        gen_sent_batch = [gen_sent[:int(s)] for gen_sent, s in zip(gen_sent_batch, caption_sizes)]
        all_gen_sents.extend(gen_sent_batch)

    bleu_score = compute_bleu_score_for_whole_dataset(all_gen_sents, filenames_to_captions)
    print("bleu_score ", bleu_score)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    run()
