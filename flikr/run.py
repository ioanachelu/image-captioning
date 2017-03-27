import tensorflow as tf
import flags
import os
from network import ShowAndTell
import os
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import time
import utils as utils
# from utils import get_caption_data, preprocess_captions, load_model, compute_bleu_score_for_batch, create_eval_json, tokenize
import eval_utils as eval_utils
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


def eval_split(sess, network, eval_kwargs):
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    language_eval = eval_kwargs.get('language_eval', True)
    dataset = eval_kwargs.get('dataset', 'data/flikr.json')

    sess.run(tf.assign(network.training, False))

    feats, captions, filenames_to_captions = utils.get_caption_data(mode=split)
    image_id_to_filename, filename_to_image_id = utils.get_filename_image_id_associations(mode=split)
    word_to_index, index_to_word, bias_init_vector, _ = utils.preprocess_captions(captions)
    maxlen = 30
    num_examples_per_epoch = len(range(0, len(feats), FLAGS.batch_size))

    # sess = tf.InteractiveSession()
    n_words = len(word_to_index)

    # loss, image, sentence, mask, generated_sentence = network.build_model()
    # global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
    # train_op = network.train(global_step, num_examples_per_epoch)
    # summary_writer, summaries = network.summary()

    # restore_var = tf.global_variables()
    # loader = tf.train.Saver(var_list=restore_var)
    # utils.load_model(loader, sess)
    # sess.run(tf.local_variables_initializer())

    predictions = []
    loss_sum = 0
    loss_evals = 0
    all_gen_sents = []
    for start, end in zip(range(0, len(feats), FLAGS.batch_size),
                          range(FLAGS.batch_size, len(feats), FLAGS.batch_size)):
        current_feats = feats[start:end]
        current_captions = captions[start:end]
        current_captions = utils.tokenize(current_captions)
        current_caption_ind = [[word_to_index[word] for word in cap if word in word_to_index] for cap in
                               current_captions]

        current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=network.n_lstm_steps)

        current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
        nonzeros = np.array([(x != 0).sum() for x in current_caption_matrix])

        for ind, row in enumerate(current_mask_matrix):
            row[:nonzeros[ind]] = 1

        batch_of_seq, loss_value = sess.run(
        [network.generated_words, network.total_loss], feed_dict={
                network.image: current_feats,
                network.sentence: current_caption_matrix,
                network.mask: current_mask_matrix,
        })
        loss_sum = loss_sum + loss_value
        loss_evals = loss_evals + 1

        batch_of_sents = utils.decode_sequence(batch_of_seq, index_to_word, current_mask_matrix, maxlen)
        batch_filenames_for_captions = filenames_to_captions[start:end]
        batch_of_filenames = [f for f, _ in batch_filenames_for_captions]
        batch_of_image_ids = [filename_to_image_id[f] for f in batch_of_filenames]

        for k, (sent, image_id) in enumerate(list(zip(batch_of_sents, batch_of_image_ids))):
            entry = {'image_id': str(image_id), 'caption': sent}
            predictions.append(entry)

    if language_eval:
        lang_stats = eval_utils.language_eval(dataset, predictions)

    # Switch back to training mode
    sess.run(tf.assign(network.training, True))
    return loss_sum / loss_evals, predictions, lang_stats


def run():
    recreate_directory_structure()
    feats, captions, filenames_to_captions = utils.get_caption_data(mode="train")
    index = np.arange(len(feats))
    np.random.shuffle(index)

    feats = feats[index]
    captions = captions[index]
    filenames_to_captions = filenames_to_captions[index]

    word_to_index, index_to_word, bias_init_vector, maxlen = utils.preprocess_captions(captions)

    # np.save('data/index_to_word', index_to_word)
    # np.save('data/word_to_index', word_to_index)
    # summary_every = len(range(0, len(feats), FLAGS.batch_size))
    # checkpoint_every = summary_every
    num_examples_per_epoch = len(range(0, len(feats), FLAGS.batch_size))

    sess = tf.InteractiveSession()
    n_words = len(word_to_index)
    # maxlen = np.max([len(x.split(' ')) for x in captions])

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
    summary_bleu = tf.Summary()

    saver = tf.train.Saver(max_to_keep=1000000)

    # restore_var = tf.global_variables()
    if FLAGS.resume:
        loader = tf.train.Saver()
        utils.load_model(loader, sess)
        sess.run(tf.local_variables_initializer())
        step = sess.run(global_step)
        start_epoch = step // num_examples_per_epoch + 1


    else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        start_epoch = 0
        step = 0

    increment_global_step = global_step.assign_add(1)

    val_result_history = {}
    best_val_score = None
    sess.run(tf.assign(network.training, True))

    for epoch in range(start_epoch, FLAGS.num_epochs):
    # while(step < FLAGS.num_steps):
    # for step in range(step_count,  FLAGS.num_steps):
        all_gen_sents = []
        for start, end in zip(range(0, len(feats), FLAGS.batch_size), range(FLAGS.batch_size, len(feats), FLAGS.batch_size)):
            start_time = time.time()
            current_feats = feats[start:end]
            current_captions = captions[start:end]
            current_captions = utils.tokenize(current_captions)
            current_caption_ind = [[word_to_index[word] for word in cap if word in word_to_index] for cap in current_captions]

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen)
            # current_caption_matrix = np.hstack(
            #     [np.full((len(current_caption_matrix), 1), 0), current_caption_matrix]).astype(int)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([(x != 0).sum() for x in current_caption_matrix])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value, summary, _, batch_of_seq = sess.run([train_op, loss, summaries, increment_global_step, generated_sentence], feed_dict={
                image: current_feats,
                sentence: current_caption_matrix,
                mask: current_mask_matrix,
                })

            batch_of_sents = utils.decode_sequence(batch_of_seq, index_to_word, current_mask_matrix, network.n_lstm_steps)
            all_gen_sents.extend(batch_of_sents)

            # if step % FLAGS.checkpoint_every == 0:
            #     saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=global_step)

            if step % FLAGS.summary_every == 0:
                print("------------------------")
                print("Reference:")
                print(' '.join(str(w) for w in current_captions[0]))
                print("Hypothesis:")
                print(batch_of_sents[0])
                print("------------------------")
                # bleu_score = utils.compute_bleu_score_for_batch(batch_of_sents, start, end, filenames_to_captions)
                # summary_bleu.value.add(tag='BLEU', simple_value=float(bleu_score))
                summary_writer.add_summary(summary, step)
                # summary_writer.add_summary(summary_bleu, step)

                duration = time.time() - start_time
                # print('Step {:d} \t loss = {:.3f}, mean_BLEU = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value,
                #                                                                                  bleu_score,
                #                                                                                  duration))
                print('Step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

                eval_kwargs = {'val_images_use': FLAGS.val_images_use,
                               'split': 'test',
                               'language_eval': FLAGS.language_eval,
                               'dataset': FLAGS.input_json}
                val_loss, predictions, lang_stats = eval_split(sess, network, eval_kwargs)

                summary = tf.Summary(value=[tf.Summary.Value(tag='validation loss', simple_value=val_loss)])
                summary_writer.add_summary(summary, step)
                for k, v in lang_stats.items():
                    summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
                    summary_writer.add_summary(summary, step)
                val_result_history[step] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if FLAGS.language_eval:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                if best_val_score is None or current_score > best_val_score:  # if true
                    best_val_score = current_score
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=global_step)
                    print("model saved to {}".format(os.path.join(FLAGS.checkpoint_dir, 'model')))
            step += 1



        # create_eval_json(all_gen_sents, filenames_to_captions, mode="train")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    run()
