import tensorflow as tf
import flags
import os
from network import ShowAndTell
import os
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import time
from utils import get_caption_data, preprocess_captions, load_model
FLAGS = tf.app.flags.FLAGS


def recreate_directory_structure():
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
            tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    if not tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
            tf.gfile.MakeDirs(FLAGS.summaries_dir)


def run():
    recreate_directory_structure()
    feats, captions = get_caption_data(mode="train")
    word_to_index, index_to_word, bias_init_vector = preprocess_captions(captions)

    np.save('data/index_to_word', index_to_word)
    print(len(range(0, len(feats), FLAGS.batch_size)))


    sess = tf.InteractiveSession()
    n_words = len(word_to_index)
    maxlen = np.max([len(x.split(' ')) for x in captions])

    network = ShowAndTell(
        image_embedding_size=FLAGS.image_embedding_size,
        num_lstm_units=FLAGS.num_lstm_units,
        embedding_size=FLAGS.embedding_size,
        batch_size=FLAGS.batch_size,
        n_lstm_steps=maxlen + 2,
        n_words=n_words,
        bias_init_vector=bias_init_vector)

    loss, image, sentence, mask = network.build_model()
    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    train_op = network.train(global_step)
    summary_writer, summaries = network.summary()

    saver = tf.train.Saver(max_to_keep=50)

    restore_var = tf.global_variables()
    if FLAGS.resume:
        loader = tf.train.Saver(var_list=restore_var)
        load_model(loader, sess)
        sess.run(tf.local_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    increment_global_step = global_step.assign_add(1)
    step_count = 0
    for epoch in range(FLAGS.num_epochs):
        start_time = time.time()
        for start, end in zip(range(0, len(feats), FLAGS.batch_size), range(FLAGS.batch_size, len(feats), FLAGS.batch_size)):
            current_feats = feats[start:end]
            current_captions = captions[start:end]
            current_caption_ind = [[word_to_index[word] for word in cap.lower().split(' ')[:-1] if word in word_to_index] for cap in current_captions]

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen + 1)
            current_caption_matrix = np.hstack(
                [np.full((len(current_caption_matrix), 1), 0), current_caption_matrix]).astype(int)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([(x != 0).sum() + 2 for x in current_caption_matrix])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value, summary, _ = sess.run([train_op, loss, summaries, increment_global_step], feed_dict={
                image: current_feats,
                sentence: current_caption_matrix,
                mask: current_mask_matrix,
                })
            if step_count % FLAGS.checkpoint_every == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=global_step)
            if step_count % FLAGS.summary_every == 0:
                summary_writer.add_summary(summary, step_count)

            step_count += 1

        duration = time.time() - start_time
        print('Epoch {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(epoch, loss_value, duration))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    run()