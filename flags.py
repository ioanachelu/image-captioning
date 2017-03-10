import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_integer('checkpoint_every', 500,
                            """Checkpoint interval""")
tf.app.flags.DEFINE_integer('summary_every', 5000,
                            """Summary interval""")
tf.app.flags.DEFINE_integer('test_summary_every', 1,
                            """Summary interval""")
tf.app.flags.DEFINE_string('GPU', "0",
                           """The GPU device to run on""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train or test""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_steps', 70000,
                            """Num of steps to train the network""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('test_summaries_dir', './test_summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('train_image_dir', '/media/ioana/turi1/image_captioning/mscoco/train2014',
                           """Directory where the training images are""")
tf.app.flags.DEFINE_string('val_image_dir', '/media/ioana/turi1/image_captioning/mscoco/val2014',
                           """Directory where the validation images are""")
tf.app.flags.DEFINE_string('train_captions_file', '/media/ioana/turi1/image_captioning/mscoco/captions_train2014.json',
                           """File where the training captions reside""")
tf.app.flags.DEFINE_string('val_captions_file', '/media/ioana/turi1/image_captioning/mscoco/captions_val2014.json',
                           """File where the validation captions reside""")
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "./mscoco/word_counts.txt",
                       "Output vocabulary file of word counts.")
tf.app.flags.DEFINE_string('pretrained_weights', './vgg16.npy',
                           """Path to where the pretrained  weights for VGG 16 reside""")
tf.app.flags.DEFINE_float('lr', 2.0, """Learning rate""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """Moving average decay for loss""")
tf.app.flags.DEFINE_integer('stepsize', 8, """Learning rate num steps per decay""")
tf.app.flags.DEFINE_float('gamma', 0.5, """Learning rate power for decay""")
tf.app.flags.DEFINE_float('momentum', 0.9, """Learning rate momentum""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, """Weight decay for weights not biases""")
tf.app.flags.DEFINE_string('test_load_queue_path', "test_queue.npy", """Path to test queue""")
tf.app.flags.DEFINE_boolean('data_augmentation', False, """Whether to use data augmentation or not""")
tf.flags.DEFINE_integer("resize_height", 346, """resize_height""")
tf.flags.DEFINE_integer("resize_width", 346, """resize_width""")
tf.flags.DEFINE_integer("height", 299, """height""")
tf.flags.DEFINE_integer("width", 299, """width""")
tf.flags.DEFINE_integer("embedding_size", 512, """embedding_size""")
tf.flags.DEFINE_integer("num_lstm_units", 512, """num_lstm_units""")
tf.flags.DEFINE_integer("vocab_size", 12000, """vocab_size""")
tf.flags.DEFINE_float("lstm_dropout_keep_prob", 0.7, """lstm_dropout_keep_prob""")

