import json
import os
import tensorflow as tf
import numpy as np
import h5py
import string
from random import shuffle, seed
from scipy.misc import imread, imresize
import flags
from collections import Counter
FLAGS = tf.app.flags.FLAGS

def preprocess_mscoco_data():
    val = json.load(open('annotations/captions_val2014.json', 'r'))
    train = json.load(open('annotations/captions_train2014.json', 'r'))

    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    # for efficiency lets group annotations by image
    itoa = {}
    for a in annots:
        imgid = a['image_id']
        if not imgid in itoa: itoa[imgid] = []
        itoa[imgid].append(a)

    # create the json blob
    out = []
    for i, img in enumerate(imgs):
        imgid = img['id']

        # coco specific here, they store train/val images separately
        loc = 'train2014' if 'train' in img['file_name'] else 'val2014'

        jimg = {}
        jimg['file_path'] = os.path.join(loc, img['file_name'])
        jimg['id'] = imgid

        sents = []
        annotsi = itoa[imgid]
        for a in annotsi:
            sents.append(a['caption'])
        jimg['captions'] = sents
        out.append(jimg)

    json.dump(out, open(FLAGS.input_json, 'w'))


def tokenize(imgs):
    for i, img in enumerate(imgs):
        img['processed_tokens'] = []
        for j, s in enumerate(img['captions']):
            txt = str(s).lower().translate(string.punctuation).strip().split()
            img['processed_tokens'].append(txt)


def build_vocab(imgs):
    captions = []
    for i, img in enumerate(imgs):
        captions.extend(img['processed_tokens'])

    counter = Counter()
    for caption in captions:
        counter.update(caption)

    print("Nb of words {}".format(len(counter)))

    common_words = [word for word in counter.items() if word[1] >= FLAGS.min_word_count]

    print("Nb of common words {}".format(len(common_words)))

    sent_lengths = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)

    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab = [word_count[0] for word_count in common_words]
    vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for txt in img['processed_tokens']:
            caption = [w if counter.get(w) > FLAGS.min_word_count else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def split_train_test(imgs):
    for i, img in enumerate(imgs):
        if i < 5000:
            img['split'] = 'val'
        elif i < 1000:
            img['split'] = 'test'
        else:
            img['split'] = 'train'

    print('assigned 5000 to val, 5000 to test.')


def encode_captions(imgs, word_to_index):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    no_images = len(imgs)
    no_captions = sum(len(img['final_captions']) for img in imgs)

    label_arrays = []
    label_start_ix = np.zeros(no_images, dtype='uint32')
    label_end_ix = np.zeros(no_images, dtype='uint32')
    label_length = np.zeros(no_captions, dtype='uint32')

    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, FLAGS.max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(FLAGS.max_length, len(s))
            caption_counter += 1
            for k, w in enumerate(s):
                if k < FLAGS.max_length:
                    Li[j, k] = word_to_index[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == no_captions, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length


def create_dataset():
    imgs = json.load(open(FLAGS.input_json, 'r'))
    seed(123)  # make reproducible
    shuffle(imgs)  # shuffle the order

    tokenize(imgs)
    vocab = build_vocab(imgs)

    index_to_word = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    word_to_index = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    split_train_test(imgs)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, word_to_index)

    # create output h5 file
    N = len(imgs)
    f = h5py.File(FLAGS.output_h5, "w")
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    dset = f.create_dataset("images", (N, 3, 256, 256), dtype='uint8') # space for resized images
    for i, img in enumerate(imgs):
        # load the image
        I = imread(img['file_path'])
        try:
            Ir = imresize(I, (256,256))
        except:
            print('failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],))
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:,:,np.newaxis]
            Ir = np.concatenate((Ir,Ir,Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2,0,1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    f.close()
    print('wrote ', FLAGS.output_h5)

    # create output json file
    out = {}
    out['index_to_word'] = index_to_word  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):
        jimg = {}
        jimg['split'] = img['split']
        if 'file_path' in img:
            jimg['file_path'] = img['file_path']  # copy it over, might need
        if 'id' in img:
            jimg['id'] = img['id']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)

    json.dump(out, open(FLAGS.output_json, 'w'))
    print('wrote ', FLAGS.output_json)


if __name__ == "__main__":
    preprocess_mscoco_data()
    create_dataset()
