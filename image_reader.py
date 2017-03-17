import tensorflow as tf
import numpy as np
import pickle
import os

FLAGS = tf.app.flags.FLAGS
IMG_MEAN = np.array((98, 97, 101), dtype=np.float32)

def prefetch_input_data(reader,
                        is_training,
                        batch_size,
                        values_per_tfrec,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        tfrec_queue_name="filename_queue",
                        value_queue_name="input_queue"):
    data_files = os.listdir("./mscoco/dataset")
    data_files = [os.path.join("./mscoco/dataset", img) for img in data_files]
    if is_training:
        data_files = [img for img in data_files if "train" in img]
    else:
        data_files = [img for img in data_files if "test" in img]

    if not data_files:
        tf.logging.fatal("Found no input files matching")
    else:
        tf.logging.info("Prefetching values from %d files", len(data_files))

    if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name=tfrec_queue_name)
        min_queue_examples = values_per_tfrec * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name)
    else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, name=tfrec_queue_name)
        capacity = values_per_tfrec + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))
    tf.summary.scalar(
        "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

    return values_queue


def parse_sequence_example(serialized):
    context, sequence = tf.parse_single_sequence_example(
        serialized,
        context_features={
            "image/data": tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            "image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

    encoded_image = context["image/data"]
    caption = sequence["image/caption_ids"]
    return encoded_image, caption


def process_image(encoded_image, is_training, thread_id=0):
    # Helper function to log an image summary to the visualizer. Summaries are
    # only logged in thread 0.
    def image_summary(name, image):
        if thread_id == 0:
            tf.summary.image(name, tf.expand_dims(image, 0))

    # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
    try:
        with tf.name_scope("decode", values=[encoded_image]):
            image = tf.image.decode_jpeg(encoded_image, channels=3)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image)
        return
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image_summary("original_image", image)

    # Resize image.
    image = tf.image.resize_images(image,
                                   size=[FLAGS.resize_height, FLAGS.resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

    # Crop to final dimensions.
    if is_training:
        image = tf.random_crop(image, [FLAGS.height, FLAGS.width, 3])
    else:
        # Central crop, assuming resize_height > height, resize_width > width.
        image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.height, FLAGS.width)

    image_summary("resized_image", image)

    # Randomly distort the image.
    # if is_training:
    #     image = distort_image(image, thread_id)

    image_summary("final_image", image)

    # Rescale to [-1,1] instead of [0, 1]
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
    """Batches input images and captions.
    This function splits the caption into an input sequence and a target sequence,
    where the target sequence is the input sequence right-shifted by 1. Input and
    target sequences are batched and padded up to the maximum length of sequences
    in the batch. A mask is created to distinguish real words from padding words.
    Example:
      Actual captions in the batch ('-' denotes padded character):
        [
          [ 1 2 5 4 5 ],
          [ 1 2 3 4 - ],
          [ 1 2 3 - - ],
        ]
      input_seqs:
        [
          [ 1 2 3 4 ],
          [ 1 2 3 - ],
          [ 1 2 - - ],
        ]
      target_seqs:
        [
          [ 2 3 4 5 ],
          [ 2 3 4 - ],
          [ 2 3 - - ],
        ]
      mask:
        [
          [ 1 1 1 1 ],
          [ 1 1 1 0 ],
          [ 1 1 0 0 ],
        ]
    Args:
      images_and_captions: A list of pairs [image, caption], where image is a
        Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
        any length. Each pair will be processed and added to the queue in a
        separate thread.
      batch_size: Batch size.
      queue_capacity: Queue capacity.
      add_summaries: If true, add caption length summaries.
    Returns:
      images: A Tensor of shape [batch_size, height, width, channels].
      input_seqs: An int32 Tensor of shape [batch_size, padded_length].
      target_seqs: An int32 Tensor of shape [batch_size, padded_length].
      mask: An int32 0/1 Tensor of shape [batch_size, padded_length].
    """
    enqueue_list = []
    for image, caption in images_and_captions:
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        enqueue_list.append([image, input_seq, target_seq, indicator])

    images, input_seqs, target_seqs, mask = tf.train.batch_join(
        enqueue_list,
        batch_size=batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch_and_pad")

    if add_summaries:
        lengths = tf.add(tf.reduce_sum(mask, 1), 1)
        tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths))
        tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths))
        tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths))

    return images, input_seqs, target_seqs, mask
