import json
import numpy as np
import tensorflow as tf
import time
import os
from model import ShowAndTell
from dataloader import DataLoader
import eval_utils
import flags
import h5py
import utils
FLAGS = tf.app.flags.FLAGS
NUM_THREADS = 2


# Setup the model
def eval():
    loader = DataLoader()
    vocab_size = loader.vocab_size
    seq_length = loader.seq_length

    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        model = ShowAndTell(vocab_size, seq_length, sess)
        model.build_model()
        model.build_generator()
        model.build_bs_generator()
        model.summary_saver()

        # Initilize the variables
        sess.run(tf.global_variables_initializer())

        # Load the model checkpoint to evaluate
        utils.load_model(model.saver, sess)
        sess.run(tf.local_variables_initializer())

        # Set sample options
        eval_kwargs = {'test_images_use': FLAGS.test_images_use,
                       'split': 'test',
                       'language_eval': FLAGS.language_eval,
                       'dataset': FLAGS.output_json}

        loss, predictions, lang_stats = eval_utils.test_eval(sess, model, loader, eval_kwargs)

    print('loss: ', loss)
    if lang_stats:
      print(lang_stats)


    json.dump(predictions, open('data/predictions.json', 'w'))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    eval()