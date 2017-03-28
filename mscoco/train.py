from dataloader import *
import numpy as np
import tensorflow as tf
import os
import json
import utils
import eval_utils
from model import ShowAndTellRaw
import flags
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


def train():
    recreate_directory_structure()

    loader = DataLoader()
    vocab_size = loader.vocab_size
    seq_length = loader.seq_length
    model = ShowAndTell(vocab_size, seq_length)

    global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)

    network.build_model()
    network.train(global_step, num_examples_per_epoch)
    network.summary()

    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads = NUM_THREADS
    tf_config.gpu_options.allow_growth = True

    saver = tf.train.Saver(restore_var, max_to_keep=1000000)

    increment_global_step = global_step.assign_add(1)
    # val_result_history = {}
    # best_val_score = None
    restore_var = tf.trainable_variables()

    with tf.Session(config=tf_config) as sess:
        if FLAGS.resume:
            model_loader = tf.train.Saver(restore_var)
            utils.load_model(model_loader, sess)
            sess.run(tf.local_variables_initializer())
            step = sess.run(global_step)
            # start_epoch = step // num_examples_per_epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # start_epoch = 0
            step = 0

        # set learning_rate
        # Assign the learning rate
        if epoch > FLAGS.learning_rate_decay_start and FLAGS.learning_rate_decay_start >= 0:
            frac = (epoch - FLAGS.learning_rate_decay_start) / FLAGS.learning_rate_decay_every
            decay_factor = FLAGS.learning_rate_decay_factor ** frac
            sess.run(tf.assign(model.lr, FLAGS.learning_rate * decay_factor))  # set the decayed rate
            sess.run(tf.assign(model.cnn_lr, FLAGS.cnn_learning_rate * decay_factor))
        else:
            sess.run(tf.assign(model.lr, FLAGS.learning_rate))
            sess.run(tf.assign(model.cnn_lr, FLAGS.cnn_learning_rate))

        # Assure in training mode
        sess.run(tf.assign(model.training, True))
        sess.run(tf.assign(model.cnn_training, True))

        while True:
            start_time = time.time()
            # Load data from train split (0)
            data = loader.get_batch('train')
            print('Read data:', time.time() - start_time)

            start_time = time.time()
            feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}

            if step <= FLAGS.finetune_cnn_after or FLAGS.finetune_cnn_after == -1:
                train_loss, merged, _, _, batch_of_seq = sess.run(
                    [model.loss, model.summaries, model.train_op, increment_global_step, generated_sentence], feed)
            else:
                # Finetune the cnn
                train_loss, merged, _, _, _, batch_of_seq = sess.run(
                    [model.loss, model.summaries, model.train_op, model.cnn_train_op, increment_global_step,
                     generated_sentence], feed)
            end_time = time.time()

            print("Step {:d} train_loss = {:.3f}, time/batch = {:.3f} sec/step" \
                  .format(step, train_loss, end_time - start_time))

            if step % FLAGS.summary_every == 0:
                model.summary_writer.add_summary(merged, step)
                model.summary_writer.flush()

                # eval_kwargs = {'val_images_use': FLAGS.val_images_use,
                #                'split': 'test',
                #                'language_eval': FLAGS.language_eval,
                #                'dataset': FLAGS.input_json}
                # val_loss, predictions, lang_stats = eval_split(sess, network, eval_kwargs)
                #
                # summary = tf.Summary(value=[tf.Summary.Value(tag='validation loss', simple_value=val_loss)])
                # summary_writer.add_summary(summary, step)
                # for k, v in lang_stats.items():
                #     summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
                #     summary_writer.add_summary(summary, step)
                # val_result_history[step] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
                #
                # # Save model if is improving on validation result
                # if FLAGS.language_eval:
                #     current_score = lang_stats['CIDEr']
                # else:
                #     current_score = - val_loss
                #
                # if best_val_score is None or current_score > best_val_score:  # if true
                #     best_val_score = current_score
                #     saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=global_step)
                #     print("model saved to {}".format(os.path.join(FLAGS.checkpoint_dir, 'model')))

            step += 1

            # Stop if reaching max epochs
            if step >= FLAGS.num_steps and FLAGS.num_steps != -1:
                break


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    train()
