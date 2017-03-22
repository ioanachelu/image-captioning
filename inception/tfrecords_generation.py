import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
import os
import threading
import sys
import numpy as np
import random
import pickle
from datetime import datetime
from inception.image_decoder import ImageDecoder

IMG_MEAN = np.array((98, 97, 101), dtype=np.float32)

def create_TFRecords(name, tf_record_batch):
    num_threads = 4
    dataset = pickle.load(open("./mscoco/" + name + ".pkl", "rb"))

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(dataset)

    spacing = np.linspace(0, len(dataset), num_threads + 1).astype(np.int)
    ranges = []
    threads = []

    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, dataset, tf_record_batch, decoder)
        t = threading.Thread(target=process_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(dataset), name))


def process_batch(thread_index, ranges, name, dataset, tf_record_batch, decoder):
    num_threads = len(ranges)
    assert not tf_record_batch % num_threads
    num_tfrecbatches_per_batch = int(tf_record_batch / num_threads)
    tfrecbatch_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                                    num_tfrecbatches_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_tfrecbatches_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        tfrecbatch = thread_index * num_tfrecbatches_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, tfrecbatch, tf_record_batch)
        output_file = os.path.join('./mscoco/dataset', output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        tfrecbatch_counter = 0
        dataset_intfrecbatch = np.arange(tfrecbatch_ranges[s], tfrecbatch_ranges[s + 1], dtype=int)
        for i in dataset_intfrecbatch:
            dataset_example = dataset[i]

            sequence_example = generate_sequence(dataset_example, name, decoder)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                tfrecbatch_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, tfrecbatch_counter, output_file))
        sys.stdout.flush()
        tfrecbatch_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_tfrecbatches_per_batch))
    sys.stdout.flush()


def generate_sequence(dataset_example, name, decoder):
    id, image, caption, caption_ids = dataset_example

    with tf.gfile.FastGFile(image, "r") as f:
        encoded_image = f.read()

    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image)
        return

    # img = tf.image.decode_jpeg(img_contents, channels=3)
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # img = tf.image.resize_images(img,
    #                              size=[FLAGS.resize_height, FLAGS.resize_width],
    #                              method=tf.image.ResizeMethod.BILINEAR)
    # if name == "train":
    #     img = tf.random_crop(img, [FLAGS.height, FLAGS.width, 3])
    # else:
    #     img = tf.image.resize_image_with_crop_or_pad(img, FLAGS.height, FLAGS.width)
    # img.set_shape([FLAGS.height, FLAGS.width, 3])
    # # img_r, img_g, img_b = tf.split(num_or_size_splits=3, value=img, axis=2)
    # # img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # img = tf.cast(img, dtype=tf.float32)
    # # Extract mean.
    # img -= IMG_MEAN

    context = tf.train.Features(feature={
        "image/image_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[id])),
        "image/data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
    })


    feature_lists = tf.train.FeatureLists(feature_list={
        "image/caption": tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(v)])) for v in caption]),
        "image/caption_ids": tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[v])) for v in caption_ids])
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example

def generate_dataset():
    create_TFRecords("train", FLAGS.train_tf_records_batch)
    create_TFRecords("test", FLAGS.test_tf_records_batch)


if __name__ == '__main__':
    generate_dataset()
