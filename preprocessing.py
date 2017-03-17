import tensorflow as tf
import flags
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS
from PIL import Image, ImageOps
import multiprocessing
import concurrent.futures
import sys
from collections import Counter
import json
import os
import threading
import sys
import nltk.tokenize
import numpy as np
import random
import pickle
from datetime import datetime
from image_decoder import ImageDecoder

IMG_MEAN = np.array((98, 97, 101), dtype=np.float32)

class Vocabulary():
    def __init__(self, vocab_dict, unk_id):
        self.vocab = vocab_dict
        self.unk_id = unk_id

    def get_id(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

class Preprocesser():
    def run(self):
        mscoco_train_dataset = self.preprocess_data(FLAGS.train_image_dir, FLAGS.train_captions_file)
        mscoco_val_dataset = self.preprocess_data(FLAGS.val_image_dir, FLAGS.val_captions_file)

        # create vocabulary from training captions
        vocab = self.create_vocab(mscoco_train_dataset)

        spreadout_train_dataset = [[tuples[0], tuples[1], caption, [vocab.get_id(word) for word in caption]] for tuples in mscoco_train_dataset for caption in tuples[2]]
        spreadout_test_dataset = [[tuples[0], tuples[1], caption, [vocab.get_id(word) for word in caption]] for tuples in mscoco_val_dataset for caption in
                                   tuples[2]]
        print("finished preprocessing")
        print("started saving the preprocessed data")
        pickle.dump(spreadout_train_dataset, open("./mscoco/train.pkl", 'wb'))
        # np.save("./mscoco/train.npy", np.asarray(spreadout_train_dataset))
        # np.save("./mscoco/test.npy", np.asarray(spreadout_test_dataset))
        pickle.dump(spreadout_test_dataset, open("./mscoco/test.pkl", 'wb'))
        print("end saving the preprocessed data")

    def preprocess_data(self, images_dir, captions_file):
        with open(captions_file, "r") as f:
            captions_data = json.load(f)

        image_dict = {}
        for image in captions_data["images"]:
            image_dict[image["id"]] = image["file_name"]
        caption_dict = {}
        for ann in captions_data["annotations"]:
            caption_dict.setdefault(ann["image_id"], [])
            caption_dict[ann["image_id"]].append(ann["caption"])

        image_captions = []
        for id in image_dict.keys():
            processed_captions = self.preprocess_captions(caption_dict[id])
            image_captions.append((id, os.path.join(images_dir, image_dict[id]), processed_captions))

        print("Finished preprocessing captions")

        return image_captions

    def preprocess_captions(self, unprocessed_captions):
        processed_captions = []
        for unprocessed_caption in unprocessed_captions:
            processed_caption = [FLAGS.start_word]
            processed_caption.extend(nltk.tokenize.word_tokenize(unprocessed_caption.lower()))
            processed_caption.append(FLAGS.end_word)
            processed_captions.append(processed_caption)

        return processed_captions

    def create_vocab(self, mscoco_train_dataset):
        print("Creating vocabulary")

        training_captions = [caption for tuple in mscoco_train_dataset for caption in tuple[2]]

        counter = Counter()
        for caption in training_captions:
            counter.update(caption)

        print("Nb of words {}".format(len(counter)))

        # take only common words

        common_words = [word for word in counter.items() if word[1] > FLAGS.min_word_count]

        print("Nb of common words {}".format(len(common_words)))

        with open(FLAGS.word_counts_output_file, "w") as f:
            for word, count in common_words:
                f.write("{} {}\n".format(word, count))
        print("Finished writing word count vocabulary file")

        # create vocab
        unk_id = len(common_words)
        vocab_dict = dict([(word, id) for (id, word) in enumerate([word_count[0] for word_count in common_words])])

        vocab = Vocabulary(vocab_dict, unk_id)

        return vocab


def run():
    preprocesser = Preprocesser()
    preprocesser.run()

if __name__ == '__main__':
    run()
