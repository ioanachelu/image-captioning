import os

import tensorflow as tf
from model import ImageCaptioningModel

from inception.preprocessing import Preprocesser

FLAGS = tf.app.flags.FLAGS


def recreate_directory_structure():
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
            tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    if not tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
    else:
        if not FLAGS.resume and FLAGS.train:
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
            tf.gfile.MakeDirs(FLAGS.summaries_dir)

    if not tf.gfile.Exists("./mscoco/train.pkl") or \
            not tf.gfile.Exists("./mscoco/test.pkl"):
        preprocesser = Preprocesser()
        preprocesser.run()

def run():
    recreate_directory_structure()
    g = tf.Graph()
    with g.as_default():
        # Create queue coordinator.
        coord = tf.train.Coordinator()

        net = ImageCaptioningModel()

        net.setup()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=g, config=config)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        net.train(sess, g)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    run()