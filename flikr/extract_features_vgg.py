import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import pandas as pd
import os
from PIL import Image
import cv2
import flags
import tensorflow as tf
from keras.models import model_from_json
FLAGS = tf.app.flags.FLAGS

batch_size = 16
img_width, img_height = 224, 224


def get_or_build_model():
    # if new or not os.path.exists(FLAGS.model_path):
    model = applications.VGG16(include_top=True, weights='imagenet')
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    for layer in model.layers:
        layer.trainable = False
        # model_json = model.to_json()
        # with open(FLAGS.model_path, "w") as json_file:
        #     json_file.write(model_json)
    # else:
    #     json_file = open(FLAGS.model_path, "r")
    #     loaded_model_json = json_file.read()
    #     json_file.close()
    #     model = model_from_json(loaded_model_json)
    return model


def extract_features_from_image(image_batch):
    # if not os.path.exists("image_feats.npy"):
    model = get_or_build_model()
    feats = model.predict(image_batch, batch_size=1)
        # np.save("image_feats.npy", feats)
    # else:
    #     feats = np.load("image_feats.npy")
    return feats


def preprocess_images(model, image_list, img_width, img_height, batch_size):
    all_feats = np.zeros([len(image_list)] + [4096])
    iter_until = len(image_list) + batch_size

    for start, end in zip(range(0, iter_until, batch_size),
                          range(batch_size, iter_until, batch_size)):
        image_batch_file = image_list[start:end]
        image_batch = [crop_image(x, target_width=img_width, target_height=img_height) for x in image_batch_file]
        image_batch = np.asarray(image_batch)
        feats = model.predict(image_batch, batch_size=batch_size)
        all_feats[start:end] = feats

    return all_feats


def crop_image(x, target_height=227, target_width=227, as_float=True):
    image = Image.open(x)
    if as_float:
        image = np.asarray(image).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))


def save_bottlebeck_features():
    # build the VGG16 network
    model = get_or_build_model()

    annotations = pd.read_table(FLAGS.annotation_path, sep='\t', header=None, names=['image', 'caption'])
    annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
    annotations['image'] = annotations['image'].map(lambda x: os.path.join(FLAGS.flickr_image_path, x.split('#')[0]))

    flick_images = annotations['image'].values
    vgg_feats = preprocess_images(model, flick_images, img_width, img_height, batch_size)

    np.save(FLAGS.feat_path, vgg_feats)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    save_bottlebeck_features()
