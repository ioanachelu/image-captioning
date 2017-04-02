import tensorflow as tf

tf.app.flags.DEFINE_integer('checkpoint_every', 2000,
                            """Checkpoint interval""")
tf.app.flags.DEFINE_integer('summary_every', 50,
                            """Summary interval""")
tf.app.flags.DEFINE_integer('test_summary_every', 1,
                            """Summary interval""")
tf.app.flags.DEFINE_string('GPU', "0",
                           """The GPU device to run on""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Resume training from latest checkpoint""")
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train or test""")
tf.app.flags.DEFINE_boolean('train_embedding_network', False,
                            """Whether to train embedding network""")
tf.app.flags.DEFINE_string('checkpoint_dir', './models/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_steps', -1,
                            """Num of steps to train the network. -1 train until manually stopped""")
tf.app.flags.DEFINE_string('summaries_dir', './summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('test_summaries_dir', './test_summaries/',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_string('train_image_dir', './train2014',
                           """Directory where the training images are""")
tf.app.flags.DEFINE_string('output_h5', './data/coco.h5',
                           """File containing the entire dataset""")
tf.app.flags.DEFINE_string('output_json', './data/coco.json',
                           """File containing the vocabulary -index to word dict- and for each image id, filename,
                            train-val-test""")
tf.app.flags.DEFINE_string('val_image_dir', './val2014',
                           """Directory where the validation images are""")
tf.app.flags.DEFINE_string('train_captions_file', './annotations/captions_train2014.json',
                           """File where the training captions reside""")
tf.app.flags.DEFINE_string('val_captions_file', './annotations/captions_val2014.json',
                           """File where the validation captions reside""")
tf.flags.DEFINE_integer("min_word_count", 5,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.app.flags.DEFINE_string('input_json', './data/coco_raw.json',
                           """File containing for each image id, filename, captions""")
tf.app.flags.DEFINE_string('pretrained_weights', './data/vgg16.npy',
                           """Path to where the pretrained  weights for VGG 16 reside""")
tf.app.flags.DEFINE_float('learning_rate', 4e-4, """Initial learning rate for the RNN""")
tf.app.flags.DEFINE_float('cnn_learning_rate', 1e-5, """Initial learning rate for the CNN""")
tf.app.flags.DEFINE_float('cnn_weight_decay', 0, """Weight decay for the CNN""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, """Moving average decay for loss""")
tf.app.flags.DEFINE_integer('learning_rate_decay_start', -1,
                            """When to start decaying the learning rate. -1 do not decay""")
tf.app.flags.DEFINE_integer('learning_rate_decay_every', 10,
                            """Number of epochs until the learning rate is decayed according to the schedule""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, """Learning rate power for exponential decay using staircase""")
tf.app.flags.DEFINE_float('momentum', 0.9, """Learning rate momentum""")
tf.app.flags.DEFINE_boolean('language_eval', True, """Whether to evaluate using NLP metrics on the validation set""")
tf.flags.DEFINE_integer("val_images_use", 3200, """How many images from the validation set to use to compute val loss and NLP metrics while training""")
tf.flags.DEFINE_integer("test_images_use", 3200, """How many image to evaluate the trained models on""")
tf.flags.DEFINE_integer("finetune_cnn_after", -1, """Start fine tunning the CNN network after a number of epochs. -1 do not finetune""")
tf.flags.DEFINE_integer("max_length", 16, """Maximum allowed length of a sequence. Even though there are longer sequences in the training set, we crop them to a this length.""")
tf.flags.DEFINE_integer("embedding_size", 512, """The embedding size used for the image embedding and the word embeddings""")
tf.flags.DEFINE_integer("num_lstm_units", 512, """Number of units in the LSTM""")
tf.flags.DEFINE_float("lstm_dropout_keep_prob", 0.5, """Keep probability used for dropout for the output of the LSTM""")
tf.app.flags.DEFINE_float('gradient_clip_value', 5.0, """Clip gradients to this value""")
tf.app.flags.DEFINE_float('initializer_scale', 0.1,
                          """Interval for uniform distributions of weights used to initialize the fully connected layer of the output logits""")
tf.flags.DEFINE_integer("beam_search_size", 2, """The size of the candidates for beam search""")

