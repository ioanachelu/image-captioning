import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS
IMG_MEAN = np.array((98, 97, 101), dtype=np.float32)


def read_labeled_image_list_from_txt(data_list):
    """Reads txt file containing paths to images and ground truth label lists.

    Args:
      data_dir: path to the directory with images and gt lists.
      data_list: path to the file with lines of the form '/path/to/image list_of_gts'.

    Returns:
      Two lists with all file names for images and label lists, respectively.
    """
    f = open(data_list, 'r')
    images = []
    labels = []
    for line in f:
        list_all = line.strip("\n").split(' ')
        image = list_all[0]
        labels_list = list_all[1:]
        labels_list = [int(label) for label in labels_list if label != '']
        labels_list = labels_list[23: 1280 - 24]
        images.append(image)
        labels.append(labels_list)
    return images, labels

def read_labeled_image_list_from_npy(data_npy):
    dataset = np.load(data_npy)
    dataset = dataset[()]
    images = []
    captions = []
    print("Dataset consists of {} images and captions".format(len(dataset)))
    for id, image, caption in dataset:
        images.append(image)
        captions.append(caption)
    return images, captions

def read_images_from_disk(input_queue):
    """Read one image and its corresponding gts with pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its gts.

    Returns:
      Two tensors: the decoded image and its gts.
    """

    img_contents = tf.read_file(input_queue[0])
    caption = input_queue[1]

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img,
                                   size=[FLAGS.resize_height, FLAGS.resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)
    img.set_shape([FLAGS.resize_height, FLAGS.resize_width, 3])
    # img_r, img_g, img_b = tf.split(num_or_size_splits=3, value=img, axis=2)
    # img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    img = tf.cast(img, dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    return img, caption





class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding gt obstacles
       from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_npy, shuffle, coord):
        '''Initialise an ImageReader.

        Args:
          data_list: path to the file with lines of the form '/path/to/image list_of_gts'.
          coord: TensorFlow queue coordinator.
        '''
        self.data_npy = data_npy
        self.coord = coord

        self.image_list, self.caption_list = read_labeled_image_list_from_npy(self.data_npy)
        self.nb_samples = len(self.image_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.captions = tf.convert_to_tensor(self.caption_list, dtype=tf.int32)
        self.queue = tf.train.slice_input_producer([self.images, self.captions],
                                                   shuffle=shuffle)  # not shuffling if it is val
        self.image, self.caption = read_images_from_disk(self.queue)


    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, caption_batch = tf.train.batch([self.image, self.caption],
                                                  num_elements)
        return image_batch, caption_batch
