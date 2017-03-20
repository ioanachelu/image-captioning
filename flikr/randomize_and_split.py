import numpy as np
import pandas as pd
import tensorflow as tf
import os
import flags
FLAGS = tf.app.flags.FLAGS


def randomize_and_split():
    feats = np.load(FLAGS.feat_path)
    annotations = pd.read_table(FLAGS.annotation_path, sep='\t', header=None, names=['image', 'caption'])
    captions = annotations['caption'].values

    # index = np.arange(len(feats))
    # np.random.shuffle(index)

    # feats = feats[index]
    # captions = captions[index]

    train_feats = feats[:28000]
    np.save("./data/train_feats.npy", train_feats)

    train_captions = captions[:28000]
    np.save("./data/train_captions.npy", train_captions)

    val_feats = feats[28000:29000]
    np.save("./data/val_feats.npy", val_feats)

    val_captions = feats[28000:29000]
    np.save("./data/val_captions.npy", val_captions)

    test_feats = feats[29000:]
    np.save("./data/test_feats.npy", test_feats)

    test_captions = feats[29000:]
    np.save("./data/test_captions.npy", test_captions)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    randomize_and_split()