import tensorflow as tf


# Basic model parameters.
tf.app.flags.DEFINE_integer('checkpoint_every', 30783,
                            """Checkpoint interval""")
tf.app.flags.DEFINE_integer('summary_every', 30783,
                            """Summary interval""")
tf.app.flags.DEFINE_integer('test_summary_every', 30783,
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
tf.app.flags.DEFINE_integer('num_steps', 1000000,
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
tf.app.flags.DEFINE_float('initial_learning_rate', 2.0, """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum', 0.9, """Learning rate momentum""")
tf.flags.DEFINE_integer("embedding_size", 512, """embedding_size""")
tf.flags.DEFINE_integer("image_embedding_size", 2048, """embedding_network_size""")
tf.flags.DEFINE_integer("num_lstm_units", 512, """num_lstm_units""")
tf.app.flags.DEFINE_integer('min_word_count', 5, """min_word_count""")
tf.app.flags.DEFINE_integer('gradient_clip_value', 5, """clip_gradient""")
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 8,
                            """Number of epochs until the learning rate is decayed according to the schedule""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, """Learning rate power for decay""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """Moving average decay for loss""")
tf.app.flags.DEFINE_string('test_image_path', './acoustic-guitar-player.jpg',
                           """File where the test_image_path is""")
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("validate_on", "test",
                       "validate_on")
tf.app.flags.DEFINE_float('initializer_scale', 0.08, """initializer_scalle""")
tf.app.flags.DEFINE_float('lstm_dropout_keep_prob', 0.7, """lstm_dropout_keep_prob""")
