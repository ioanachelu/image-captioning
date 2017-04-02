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
from dataloader import DataLoader

FLAGS = tf.app.flags.FLAGS

def validate(step, global_step, sess, model, loader, best_val_score):
    # Do evaluation procedure and if model is better save it
    if step % FLAGS.checkpoint_every == 0:
        eval_kwargs = {'val_images_use': FLAGS.val_images_use,
                       'split': 'val',
                       'language_eval': FLAGS.language_eval,
                       'dataset': FLAGS.output_json}
        val_loss, predictions, lang_stats = eval_utils.val_eval(sess, model, loader, eval_kwargs)

        # Add summaries with validation loss and NLP metrics results
        summary = tf.Summary(value=[tf.Summary.Value(tag='validation loss', simple_value=val_loss)])
        model.summary_writer.add_summary(summary, step)
        for k, v in lang_stats.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v)])
            model.summary_writer.add_summary(summary, step)

        # Save model if is improving on validation result
        if FLAGS.language_eval:
            current_score = lang_stats['CIDEr']
        else:
            current_score = - val_loss

        # Save model if score ( loss or NLP metric CIDEr result improved on the validation set )
        if best_val_score is None or current_score > best_val_score:  # if true
            best_val_score = current_score
            model.saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'model'), global_step=global_step)
            print("model saved to {}".format(os.path.join(FLAGS.checkpoint_dir, 'model')))
