from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import tensorflow as tf
import numpy as np
import random
import skimage
import skimage.io
import scipy.misc
import flags

FLAGS = tf.app.flags.FLAGS

# Data loader class used to read the H5 file format containing the dataset with all the images
class DataLoader():
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        # load the json file which contains additional information about the dataset - like index_to_word mapping and image
        # info
        print('DataLoader loading json file: ', FLAGS.output_json)
        self.info = json.load(open(FLAGS.output_json))

        self.index_to_word = self.info['index_to_word']
        self.vocab_size = len(self.index_to_word)
        self.seq_per_img = 5 # hardcoded for MSCOCO nr of captions per image
        print('Vocab size is ', self.vocab_size)

        # Open the hdf5 file
        print('DataLoader loading h5 file: ', FLAGS.output_h5)
        self.h5_file = h5py.File(FLAGS.output_h5)

        # Extract image shape from dataset
        images_size = self.h5_file['images'].shape
        assert len(images_size) == 4, 'images should be a 4D tensor'
        assert images_size[2] == images_size[3], 'width and height must match'

        self.num_images = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_size = images_size[2]
        print('read %d images of size %dx%dx%d' % (self.num_images,
                                                   self.num_channels, self.max_image_size, self.max_image_size))

        # load in the sequence data
        seq_size = self.h5_file['captions'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.first_caption_pointer_array = self.h5_file['first_caption_pointer_array'][:]
        self.last_caption_pointer_array = self.h5_file['last_caption_pointer_array'][:]

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.index_to_word

    def get_seq_length(self):
        return self.seq_length

    def get_batch(self, split, batch_size=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size

        img_batch = np.ndarray([batch_size, 224, 224, 3], dtype='float32')
        caption_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='float32')

        max_index = len(split_ix)
        # Start a new epoch is true
        wrapped = False

        infos = []

        for i in range(batch_size):
            it = self.iterators[split]
            it_next = it + 1
            if it_next >= max_index:
                it_next = 0
                wrapped = True
            self.iterators[split] = it_next
            ix = split_ix[it]

            # fetch image
            img = self.h5_file['images'][ix, :, :, :].transpose(1, 2, 0)
            # crop center image and rescale
            img_batch[i] = img[16:240, 16:240, :].astype('float32') / 255.0

            # fetch the sequence labels
            ix1 = self.first_caption_pointer_array[ix] - 1  # first_caption_pointer_array starts from 1
            ix2 = self.last_caption_pointer_array[ix] - 1
            ncap = ix2 - ix1 + 1  # number of captions available for this image
            assert ncap > 0, 'This image doesnt have any captions'

            if ncap < self.seq_per_img:
                # we need to subsample (with replacement) bootsrap
                seq = np.zeros([self.seq_per_img, self.seq_length], dtype='int')
                for q in range(self.seq_per_img):
                    ixl = random.randint(ix1, ix2)
                    seq[q, :] = self.h5_file['captions'][ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
                seq = self.h5_file['captions'][ixl: ixl + self.seq_per_img, :self.seq_length]

            caption_batch[i * self.seq_per_img: (i + 1) * self.seq_per_img, 1: self.seq_length + 1] = seq

            # record associated info as well
            info_dict = {}
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, caption_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        data['images'] = img_batch
        data['captions'] = caption_batch
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0

