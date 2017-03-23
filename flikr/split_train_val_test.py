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
    captions_numbers = annotations['image'].map(lambda x: x.split('#')[1]).values
    image_filenames = annotations['image'].map(lambda x: os.path.join(FLAGS.flickr_image_path, x.split('#')[0])).values

    filename_caption_association = list(zip(image_filenames, captions))
    # index = np.arange(len(feats))
    # np.random.shuffle(index)

    # feats = feats[index]
    # captions = captions[index]

    train_feats = feats[:-5000]
    np.save("./data/train_feats.npy", train_feats)
    train_captions = captions[:-5000]
    np.save("./data/train_captions.npy", train_captions)
    train_filename_caption_association = filename_caption_association[:-5000]
    np.save("./data/train_filename_caption_association.npy", train_filename_caption_association)

    # val_feats = feats[28000:29000]
    # np.save("./data/val_feats.npy", val_feats)
    # val_captions = feats[28000:29000]
    # np.save("./data/val_captions.npy", val_captions)
    # val_filename_caption_association = filename_caption_association[28000:29000]
    # np.save("./data/val_filename_caption_association.npy", val_filename_caption_association)

    test_feats = feats[-5000:]
    np.save("./data/test_feats.npy", test_feats)
    test_captions = captions[-5000:]
    np.save("./data/test_captions.npy", test_captions)
    test_filename_caption_association = filename_caption_association[-5000:]
    np.save("./data/test_filename_caption_association.npy", test_filename_caption_association)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    randomize_and_split()