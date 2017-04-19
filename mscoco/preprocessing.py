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
    # Load the MSCOCO annotations
    val = json.load(open('annotations/captions_val2014.json', 'r'))
    train = json.load(open('annotations/captions_train2014.json', 'r'))

    imgs = val['images'] + train['images']
    annots = val['annotations'] + train['annotations']

    index_to_annotation = {}
    for a in annots:
        imgid = a['image_id']
        index_to_annotation.setdefault(imgid, [])
        index_to_annotation[imgid].append(a)

    # Construct json containing a list of image dictionaries containing for each image: image_id, file_path and list of
    # captions
    image_list = []
    for i, img in enumerate(imgs):
        # MSCOCO file path differ for images in  predefined train vs validation set
        image_info = {}
        image_info['file_path'] = os.path.join('train2014' if 'train' in img['file_name'] else 'val2014',
                                               img['file_name'])
        image_info['id'] = img['id']

        image_info['captions'] = [a['caption'] for a in index_to_annotation[img['id']]]
        image_list.append(image_info)

    json.dump(image_list, open(FLAGS.input_json, 'w'))


def tokenize(image_list):
    for i, image_info in enumerate(image_list):
        image_info['processed_tokens'] = [str(s).lower().translate(string.punctuation).strip().split() for s in
                                          image_info['captions']]


def build_vocab(image_list):
    captions = []
    for i, image_info in enumerate(image_list):
        captions.extend(image_info['processed_tokens'])

    counter = Counter()
    for caption in captions:
        counter.update(caption)

    print("Nb of words {}".format(len(counter)))
    # keep only the common words
    common_words = [word for word in counter.items() if word[1] >= FLAGS.min_word_count]
    print("Nb of common words {}".format(len(common_words)))

    max_len = np.max([len(c) for c in captions])
    print('Max length of a sentence: {}, but will trim them at {}'.format(max_len, FLAGS.max_length))

    # additional special UNK token we will use below to map infrequent words to
    print('Inserting the special UNK token for unknown words that did not get '
          'selected in the vocabulary - too infrequent')
    vocab = [word_count[0] for word_count in common_words]
    vocab.append('UNK')

    # replace words not in vocabulary with unknown
    for image_info in image_list:
        image_info['final_captions'] = [[w if counter.get(w) > FLAGS.min_word_count else 'UNK' for w in caption] for
                                        caption in image_info['processed_tokens']]

    return vocab


def split_train_test(image_list):
    for i, image_info in enumerate(image_list):
        if i < 5000:
            image_info['split'] = 'val'
        elif i < 10000:
            image_info['split'] = 'test'
        else:
            image_info['split'] = 'train'

    print('Keeping 5000 for validation, 5000 for testing. Train on the rest')


def encode_captions(image_list, word_to_index):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    no_images = len(image_list)
    no_captions = sum(len(image_info['final_captions']) for image_info in image_list)

    # list of caption arrays
    caption_arrays_list = []
    # array of pointers to the first caption of each image
    first_caption_pointer_array = np.zeros(no_images, dtype='uint32')
    # array of pointers to the last caption of each image
    last_caption_pointer_array = np.zeros(no_images, dtype='uint32')

    # array of length of all captions for each image
    full_captions_length_array = np.zeros(no_captions, dtype='uint32')

    caption_counter = 0
    counter = 1
    for i, image_info in enumerate(image_list):
        no_captions_per_image = len(image_info['final_captions'])
        assert no_captions_per_image > 0, 'Error: some image has no captions. Should be 5 for all.'

        # alocate space for each caption to be max_length long
        image_captions_array = np.zeros((no_captions_per_image, FLAGS.max_length), dtype='uint32')
        for j, caption in enumerate(image_info['final_captions']):
            full_captions_length_array[caption_counter] = min(FLAGS.max_length, len(caption))
            caption_counter += 1
            # load the caption words in the allocated array
            for k, w in enumerate(caption):
                if k < FLAGS.max_length:
                    image_captions_array[j, k] = word_to_index[w]

        caption_arrays_list.append(image_captions_array)
        first_caption_pointer_array[i] = counter
        last_caption_pointer_array[i] = counter + no_captions_per_image - 1

        counter += no_captions_per_image

    # put all the captions in a big array
    caption_array = np.concatenate(caption_arrays_list, axis=0)
    assert caption_array.shape[
               0] == no_captions, 'First dimension of the caption array should be the number of captions'
    assert np.all(full_captions_length_array > 0), 'All captions should have words'

    print('Encoded captions in an array with shape ', caption_array.shape)
    return caption_array, first_caption_pointer_array, last_caption_pointer_array, full_captions_length_array


def create_dataset():
    # Load the raw json file
    image_list = json.load(open(FLAGS.input_json, 'r'))
    seed(123)
    shuffle(image_list)  # shuffle with the same seed

    # preprocess captions: tokenize, split, decapitalize
    tokenize(image_list)
    # build the vocabulary dictionary
    vocab = build_vocab(image_list)

    # a dictionay indexed at 1 containing a mapping index - vocabulary word
    index_to_word = {i + 1: w for i, w in enumerate(vocab)}
    # the inverse mapping of index_to_word
    word_to_index = {w: i + 1 for i, w in enumerate(vocab)}

    # split the processed dataset into train-val-test sets and add tags for each image
    split_train_test(image_list)

    # encode captions in large arrays, ready to store in hdf5 file format
    captions_array, first_caption_pointer_array, last_caption_pointer_array, full_captions_length_array = encode_captions(
        image_list, word_to_index)

    # create output h5 file
    no_images = len(image_list)
    f = h5py.File(FLAGS.output_h5, "w")
    f.create_dataset("captions", dtype='uint32', data=captions_array)
    f.create_dataset("first_caption_pointer_array", dtype='uint32', data=first_caption_pointer_array)
    f.create_dataset("last_caption_pointer_array", dtype='uint32', data=last_caption_pointer_array)
    f.create_dataset("full_captions_length_array", dtype='uint32', data=full_captions_length_array)
    h5_dataset = f.create_dataset("images", (no_images, 3, 256, 256), dtype='uint8')
    for i, image_info in enumerate(image_list):
        # load the image
        img = imread(image_info['file_path'])
        try:
            # resize the images
            img_resized = imresize(img, (256, 256))
        except:
            print('Error: failed to resize image {}'.format(image_info['file_path']))
            raise
        # concatenate the single channel if we have grayscale images with only 1 channel
        if len(img_resized.shape) == 2:
            img_resized = img_resized[:, :, np.newaxis]
            img_resized = np.concatenate((img_resized, img_resized, img_resized), axis=2)
        # Swap the axes channel wise because that is how the VGG processes the image
        img_resized = img_resized.transpose(2, 0, 1)
        # load in the h5 dataset
        h5_dataset[i] = img_resized
        # print the progress of the process
        if i % 1000 == 0:
            print('Processing {:d}/{:d} ({:.2f}% done)'.format(i, no_images, i * 100 / no_images))
    f.close()
    print('Finished writing the H5 dataset file {}'.format(FLAGS.output_h5))

    # create output json file containing the index_to_word mapping of the vocabulary and also for each image: id,
    # filename and the split type: train, test, validation
    out = {}
    out['index_to_word'] = index_to_word
    out['images'] = []
    for i, image_info in enumerate(image_list):
        image_dict = {}
        image_dict['split'] = image_info['split']
        if 'file_path' in image_info:
            image_dict['file_path'] = image_info['file_path']
        if 'id' in image_info:
            image_dict['id'] = image_info['id']

        out['images'].append(image_dict)

    json.dump(out, open(FLAGS.output_json, 'w'))
    print('Finished writing the output json with index to word dict and image infos {}'.format(FLAGS.output_json))


if __name__ == "__main__":
    preprocess_mscoco_data()
    create_dataset()
