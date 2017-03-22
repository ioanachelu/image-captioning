import tensorflow as tf
import flags
import os
from network import ShowAndTell
import os
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from utils import preprocess_captions, get_caption_data, load_model
import time
from extract_features_vgg import crop_image, extract_features_from_image
FLAGS = tf.app.flags.FLAGS


def read_image(path):
    img = crop_image(path, target_height=224, target_width=224)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = img[None, ...]
    return img


def eval():
    image_val = read_image(FLAGS.test_image_path)
    maxlen = 30
    feats = extract_features_from_image(image_val)
    index_to_word = np.load("data/index_to_word.npy").tolist()
    n_words = len(index_to_word)

    sess = tf.InteractiveSession()

    caption_generator = ShowAndTell(
        image_embedding_size=FLAGS.image_embedding_size,
        num_lstm_units=FLAGS.num_lstm_units,
        embedding_size=FLAGS.embedding_size,
        batch_size=FLAGS.batch_size,
        n_lstm_steps=maxlen,
        n_words=n_words)

    image_embedding_placeholder, generated_words, mask = caption_generator.build_generator(maxlen=maxlen)
    all_trainable = [v for v in tf.global_variables() if 'block' not in v.name and 'fc1' not in v.name and 'fc2' not in v.name and 'predictions' not in v.name]
    # restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=all_trainable)
    load_model(loader, sess)

    generated_word_index, mask_pred = sess.run([generated_words, mask], feed_dict={image_embedding_placeholder: feats})
    generated_word_index = np.hstack(generated_word_index)
    mask_pred = np.hstack(mask_pred)

    generated_words = [index_to_word[x] for i, x in enumerate(generated_word_index) if not mask_pred[i]]
    # punctuation = np.argmax(np.array(generated_words) == '.') + 1

    # generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    print(generated_sentence)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    eval()