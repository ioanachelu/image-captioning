import tensorflow as tf
import flags
import os
from network import ShowAndTell
import os
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from utils import preprocess_captions, get_caption_data, load_model, tokenize, preprocess_for_test, get_all_captions_for_filename
import time
from extract_features_vgg import crop_image, extract_features_from_image
import nltk
import random
from PIL import Image
FLAGS = tf.app.flags.FLAGS


def read_image(path):
    img = crop_image(path, target_height=224, target_width=224)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = img[None, ...]
    return img


def eval():
    feats, captions, filenames_to_captions = get_caption_data(mode="test")
    index = random.randint(0, 1000)
    image_feats = feats[index]

    image = Image.open(filenames_to_captions[index][0])
    image.show()

    image_captions = get_all_captions_for_filename(filenames_to_captions[index][0], filenames_to_captions)
    # image_captions = captions[index]
    index_to_word = np.load("data/index_to_word.npy").tolist()
    n_words = len(index_to_word)
    maxlen = np.max([len(c) for c in tokenize(image_captions)])

    sess = tf.InteractiveSession()

    caption_generator = ShowAndTell(
        image_embedding_size=FLAGS.image_embedding_size,
        num_lstm_units=FLAGS.num_lstm_units,
        embedding_size=FLAGS.embedding_size,
        batch_size=FLAGS.batch_size,
        n_lstm_steps=maxlen,
        n_words=n_words)

    image_embedding_placeholder, generated_words, mask = caption_generator.build_generator(maxlen=maxlen)

    restore_var = tf.global_variables()
    loader = tf.train.Saver(var_list=restore_var)
    load_model(loader, sess)
    sess.run(tf.local_variables_initializer())

    image_feats = np.expand_dims(image_feats, 0)

    generated_word_index, mask_pred = sess.run([generated_words, mask], feed_dict={image_embedding_placeholder: image_feats})
    generated_word_index = np.hstack(generated_word_index)
    mask_pred = np.hstack(mask_pred)

    generated_words = [index_to_word[x] for i, x in enumerate(generated_word_index) if not mask_pred[i]]
    generated_words = [w for w in generated_words if w != FLAGS.start_word and w != FLAGS.end_word]

    generated_sentence = ' '.join(generated_words)
    reference_captions = preprocess_for_test(image_captions)
    print("------------------------")
    print("References:")
    for c in reference_captions:
        print(c)
    print("Hypothesis:")
    print(generated_sentence)
    print("------------------------")

    cc = nltk.translate.bleu_score.SmoothingFunction()
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference_captions, generated_sentence, smoothing_function=cc.method3)
    print('BLEU: ', bleu_score)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    eval()