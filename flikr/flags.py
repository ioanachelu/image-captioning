import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_integer('checkpoint_every', 1242,
                            """Checkpoint interval""")
tf.app.flags.DEFINE_integer('summary_every', 1242,
                            """Summary interval""")
tf.app.flags.DEFINE_integer('test_summary_every', 1242,
                            """Summary interval""")
tf.app.flags.DEFINE_string('GPU', "0",
                           """The GPU device to run on""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train or test""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_epochs', 1000,
                            """Num of epochs to train the network""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('test_summaries_dir', './test_summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('train_image_dir', './train2014',
                           """Directory where the training images are""")
tf.app.flags.DEFINE_string('val_image_dir', './val2014',
                           """Directory where the validation images are""")
tf.app.flags.DEFINE_string('data_path', './data',
                           """File where the data is""")
tf.app.flags.DEFINE_string('feat_path', './data/feats.npy',
                           """File where the features from vgg are""")
tf.app.flags.DEFINE_string('model_path', './data/model.json',
                           """File where the model json is""")
tf.app.flags.DEFINE_string('flickr_image_path', './images/flickr30k-images/',
                           """File where the images from flikr are""")
tf.app.flags.DEFINE_string('annotation_path', './data/results_20130124.token',
                           """Path to annotation file""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum', 0.9, """Learning rate momentum""")
tf.flags.DEFINE_integer("embedding_size", 256, """embedding_size""")
tf.flags.DEFINE_integer("image_embedding_size", 4096, """embedding_network_size""")
tf.flags.DEFINE_integer("num_lstm_units", 256, """num_lstm_units""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch', 1242, """Number of examples per epoch of training data""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 20,
                            """Number of epochs until the learning rate is decayed according to the schedule""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, """Learning rate power for decay""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """Moving average decay for loss""")
tf.app.flags.DEFINE_string('test_image_path', './acoustic-guitar-player.jpg',
                           """File where the test_image_path is""")