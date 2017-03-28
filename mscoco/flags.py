import tensorflow as tf

# Basic model parameters.
tf.app.flags.DEFINE_integer('checkpoint_every', 1,
                            """Checkpoint interval""")
tf.app.flags.DEFINE_integer('summary_every', 1,
                            """Summary interval""")
tf.app.flags.DEFINE_integer('test_summary_every', 1,
                            """Summary interval""")
tf.app.flags.DEFINE_string('GPU', "0",
                           """The GPU device to run on""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('resume', True,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train or test""")
tf.app.flags.DEFINE_boolean('train_embedding_network', False,
                            """Whether to train embedding network""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_steps', -1,
                            """Num of steps to train the network""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('test_summaries_dir', './test_summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('train_image_dir', './train2014',
                           """Directory where the training images are""")
tf.app.flags.DEFINE_string('output_h5', './data/coco.h5',
                           """Directory where the training images are""")
tf.app.flags.DEFINE_string('output_json', './data/coco.json',
                           """Directory where the training images are""")
tf.app.flags.DEFINE_string('val_image_dir', './val2014',
                           """Directory where the validation images are""")
tf.app.flags.DEFINE_string('train_captions_file', './annotations/captions_train2014.json',
                           """File where the training captions reside""")
tf.app.flags.DEFINE_string('val_captions_file', './annotations/captions_val2014.json',
                           """File where the validation captions reside""")
tf.app.flags.DEFINE_string('embedding_network_checkpoint_file', './embedding_models/inception_v3.ckpt',
                           """Checkpoint file from where to load the weights for the embedding network""")
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_integer("train_tf_records_batch", 256,
                        "Nr of TFRecords for each data file in the preprocessing step for the train dataset")
tf.flags.DEFINE_integer("val_tf_records_batch", 4,
                        "Nr of TFRecords for each data file in the preprocessing step for the val dataset")
tf.flags.DEFINE_integer("test_tf_records_batch", 8,
                        "Nr of TFRecords for each data file in the preprocessing step for the test dataset")
tf.flags.DEFINE_string("word_counts_output_file", "./mscoco/word_counts.txt",
                       "Output vocabulary file of word counts.")
tf.app.flags.DEFINE_string('input_json', './data/coco_raw.json', """""")
tf.app.flags.DEFINE_string('pretrained_weights', './data/vgg16.npy',
                           """Path to where the pretrained  weights for VGG 16 reside""")
tf.app.flags.DEFINE_float('train_embedding_network_learning_rate', 0.0005,
                          """Learning rate when training embedding network""")
tf.app.flags.DEFINE_float('learning_rate', 4e-4, """Initial learning rate""")
tf.app.flags.DEFINE_float('cnn_learning_rate', 1e-5, """Initial learning rate""")
tf.app.flags.DEFINE_float('cnn_weight_decay', 0, """Initial learning rate""")
tf.app.flags.DEFINE_float('weight_initializer_interval', 0.08,
                          """Interval for uniform distributions of weights used to initialize the network""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """Moving average decay for loss""")
tf.app.flags.DEFINE_integer('learning_rate_decay_start', -1, """Number of examples per epoch of training data""")
tf.app.flags.DEFINE_integer('learning_rate_decay_every', 10,
                            """Number of epochs until the learning rate is decayed according to the schedule""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, """Learning rate power for decay""")
tf.app.flags.DEFINE_float('momentum', 0.9, """Learning rate momentum""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, """Weight decay for weights not biases""")
tf.app.flags.DEFINE_boolean('data_augmentation', False, """Whether to use data augmentation or not""")
tf.flags.DEFINE_integer("finetune_cnn_after", -1, """finetune_cnn_after""")
tf.flags.DEFINE_integer("resize_height", 346, """resize_height""")
tf.flags.DEFINE_integer("resize_width", 346, """resize_width""")
tf.flags.DEFINE_integer("height", 299, """height""")
tf.flags.DEFINE_integer("width", 299, """width""")
tf.flags.DEFINE_integer("max_length", 16, """max_length""")
tf.flags.DEFINE_integer("input_encoding_size", 512, """input_encoding_size""")
tf.flags.DEFINE_integer("embedding_size", 512, """embedding_size""")
tf.flags.DEFINE_integer("num_lstm_units", 512, """num_lstm_units""")
tf.flags.DEFINE_integer("vocab_size", 12000, """vocab_size""")
tf.flags.DEFINE_float("lstm_dropout_keep_prob", 0.7, """lstm_dropout_keep_prob""")
tf.app.flags.DEFINE_float('gradient_clip_value', 5.0, """Clip gradients to this value""")
