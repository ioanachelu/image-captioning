from dataloader import *
import numpy as np
import tensorflow as tf
import os
import json
import utils
import eval_utils
from model import ShowAndTell
import flags
import time
import val
from dataloader import DataLoader
FLAGS = tf.app.flags.FLAGS

# Reacreate all the directories for model and summary storage if you are retraining, otherwise resume
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
    # recreate_directory_structure()

    # Initialize the data loader class
    loader = DataLoader()
    vocab_size = loader.vocab_size
    seq_length = loader.seq_length

    # Create session configuration
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads = 2
    tf_config.gpu_options.allow_growth = True

    # Variable to keep the best validation score in order to save the models that perform best on the validation set
    best_val_score = None

    # Run in session with config
    with tf.Session(config=tf_config) as sess:

        # Create model for caption generator
        model = ShowAndTell(vocab_size, seq_length, sess)
        # Keep global step to be used to save and resume models
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)

        # build model for training
        model.build_model()
        # build model for validation - sequence generation
        model.build_generator()
        # build saver and summary information
        model.summary_saver()

        # keep operation for incrementing the global step
        increment_global_step = global_step.assign_add(1)

        # Handle training from scratch vs resuming training from a previously saved model
        if FLAGS.resume:
            sess.run(tf.global_variables_initializer())
            utils.load_model(model.saver, sess)
            sess.run(tf.local_variables_initializer())
            step = sess.run(global_step)
            # start_epoch = step // num_examples_per_epoch + 1
            # epoch is not saved and cannot be recovered but it is never used anyway so we don't care
            epoch = 0
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # start_epoch = 0
            step = 0
            epoch = 0

        # set learning_rate
        sess.run(tf.assign(model.lr, FLAGS.learning_rate))
        sess.run(tf.assign(model.cnn_lr, FLAGS.cnn_learning_rate))

        # Assure in training mode - need this for dropout keep probability
        sess.run(tf.assign(model.training, True))
        sess.run(tf.assign(model.cnn_training, True))

        while True:
            start_time = time.time()
            # Load data from train split - train
            data = loader.get_batch('train')
            print('Read data took :', time.time() - start_time)

            start_time = time.time()
            feed = {model.images: data['images'], model.captions: data['captions'], model.masks: data['masks']}

            if step <= FLAGS.finetune_cnn_after or FLAGS.finetune_cnn_after == -1:
                train_loss, merged, _, _ = sess.run(
                    [model.loss, model.summaries, model.train_op, increment_global_step], feed)
            else:
                # Finetune the cnn. Not used so far
                train_loss, merged, _, _, _ = sess.run(
                    [model.loss, model.summaries, model.train_op, model.cnn_train_op, increment_global_step], feed)
            end_time = time.time()

            print("Step {:d} train_loss = {:.3f}, time/batch = {:.3f} sec/step" \
                  .format(step, train_loss, end_time - start_time))

            # Dump summary statistics to tensorflow event file
            if step % FLAGS.summary_every == 0:
                model.summary_writer.add_summary(merged, step)
                model.summary_writer.flush()

            best_val_score = val.validate(step, global_step, sess, model, loader, best_val_score)

            step += 1
            if data['bounds']['wrapped']:
                epoch += 1

            # Stop if reaching max epochs
            if step >= FLAGS.num_steps and FLAGS.num_steps != -1:
                break


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    train()
