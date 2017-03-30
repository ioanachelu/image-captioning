import numpy as np
import tensorflow as tf
import utils
import vgg

FLAGS = tf.app.flags.FLAGS
MAX_STEPS = 30

class ShowAndTell():
    def __init__(self, vocab_size, seq_length, sess):
        self.embedding_size = np.int(FLAGS.embedding_size)
        self.num_lstm_units = np.int(FLAGS.embedding_size)
        self.batch_size = np.int(FLAGS.batch_size)
        self.seq_length = np.int(seq_length)
        self.vocab_size = np.int(vocab_size)
        self.sess = sess

        # Variable indicating in training mode or evaluation mode
        self.training = tf.Variable(True, trainable=False, name="training")

        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
        self.labels = tf.placeholder(tf.int32, [None, self.seq_length + 2])
        self.masks = tf.placeholder(tf.float32, [None, self.seq_length + 2])

        self.initializer = tf.random_uniform_initializer(
            minval=-FLAGS.initializer_scale,
            maxval=FLAGS.initializer_scale)

        self.keep_prob = tf.cond(self.training,
                                 lambda: tf.constant(1 - FLAGS.lstm_dropout_keep_prob),
                                 lambda: tf.constant(1.0), name='keep_prob')

        if FLAGS.resume:
            pretrained_weights = None
        else:
            pretrained_weights = FLAGS.pretrained_weights

        self.cnn = vgg.Vgg16(pretrained_weights)
        with tf.variable_scope("cnn") as scope:
            self.cnn.build(self.images)
        self.fc7 = self.cnn.drop7
        self.cnn_training = self.cnn.training

        with tf.variable_scope("lstm"):
            with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
                self.embedding_map = tf.get_variable(
                    name="map",
                    shape=[self.vocab_size + 1, self.embedding_size],
                    initializer=self.initializer)

            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.num_lstm_units)
            self.lstm = tf.contrib.rnn.DropoutWrapper(self.lstm, 1.0, self.keep_prob)

            # self.embed_word_W = tf.get_variable(shape=[num_lstm_units, n_words], initializer=self.initializer,
            #                                     name='embed_word_W')
        with tf.variable_scope("cnn"):
            self.image_emb = tf.contrib.layers.fully_connected(
                inputs=self.fc7,
                num_outputs=self.embedding_size,
                activation_fn=None,
                scope='encode_image')

    def build_model(self):
        with tf.name_scope("batch_size"):
            # Get batch_size from the first dimension of self.images
            self.batch_size = tf.shape(self.images)[0]

        # ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, name='loss_avg')
        # self.generated_words = [tf.cast(self.labels[:, 0], dtype=tf.int64)]

        with tf.variable_scope("lstm") as lstm_scope:
            # Replicate self.seq_per_img times for each image embedding
            image_emb = tf.reshape(tf.tile(tf.expand_dims(self.image_emb, 1), [1, 5, 1]),
                                   [self.batch_size * 5, self.embedding_size])

            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.embedding_map, self.labels[:, :self.seq_length + 1])

            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=current_emb)
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]
            rnn_inputs = [image_emb] + rnn_inputs

            # The initial sate is zero
            initial_state = self.lstm.zero_state(self.batch_size * 5, tf.float32)

            outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_inputs, initial_state, self.lstm,
                                                                        loop_function=None)
            # outputs, last_state = tf.nn.dynamic_rnn(self.lstm, rnn_inputs, initial_state=initial_state)

            outputs = tf.concat(axis=0, values=outputs[1:])
            self.logits_flat = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=self.vocab_size + 1,
                activation_fn=None,
                scope='logit')
            self.logits = tf.split(axis=0, num_or_size_splits=len(rnn_inputs) - 1, value=self.outputs)



        with tf.variable_scope("loss"):
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(self.logits,
                                                                      [tf.squeeze(label, [1]) for label in
                                                                       tf.split(axis=1,
                                                                                num_or_size_splits=self.seq_length + 1,
                                                                                value=self.labels[:, 1:])],
                                                                      # self.labels[:,1:] is the target
                                                                      [tf.squeeze(mask, [1]) for mask in
                                                                       tf.split(axis=1,
                                                                                num_or_size_splits=self.seq_length + 1,
                                                                                value=self.masks[:, 1:])])
            self.loss = tf.reduce_mean(loss)

        # with tf.variable_scope("loss"):
        #     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_flat, self.labels[:, 1:])
        #     masked_losses = self.masks[:, 1:] * losses
        #     mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / self.seq_length + 1
        #     self.loss = tf.reduce_mean(mean_loss_by_example)

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.cnn_lr = tf.Variable(0.0, trainable=False)

        # Collect the rnn variables, and create the optimizer of rnn
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')
        grads = utils.clip_by_value(tf.gradients(self.loss, tvars), -FLAGS.gradient_clip_value, FLAGS.gradient_clip_value)

        for grad, weight in zip(grads, tvars):
            tf.summary.histogram(weight.name + '_grad', grad)
            tf.summary.histogram(weight.name, weight)

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.8, beta2=0.999, epsilon=1e-8)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # Collect the cnn variables, and create the optimizer of cnn
        cnn_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
        cnn_grads = utils.clip_by_value(tf.gradients(self.loss, cnn_tvars), -FLAGS.gradient_clip_value, FLAGS.gradient_clip_value)

        for grad, weight in zip(cnn_grads, cnn_tvars):
            if grad is not None and weight is not None:
                tf.summary.histogram(weight.name + '_grad', grad)
                tf.summary.histogram(weight.name, weight)

        cnn_optimizer = tf.train.AdamOptimizer(self.cnn_lr, beta1=0.8, beta2=0.999, epsilon=1e-8)
        self.cnn_train_op = cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_tvars))

        tf.summary.scalar('training loss', self.loss)
        tf.summary.scalar('learning rate', self.lr)
        tf.summary.scalar('cnn learning rate', self.cnn_lr)
        self.summaries = tf.summary.merge_all()

    def build_generator(self):
        # Variables for the sample setting
        # self.sample_max = tf.Variable(True, trainable=False, name="sample_max")

        self.generator = []
        with tf.variable_scope("lstm") as lstm_scope:
            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.embedding_map, tf.zeros([self.batch_size], tf.int32))

            rnn_inputs = [self.image_emb] + [current_emb] + [0] * (MAX_STEPS - 1)
            initial_state = self.lstm.zero_state(self.batch_size, tf.float32)

            tf.get_variable_scope().reuse_variables()

            def loop(prev, i):
                if i == 1:
                    return rnn_inputs[1]
                with tf.variable_scope(lstm_scope):
                    prev = tf.contrib.layers.fully_connected(
                        inputs=prev,
                        num_outputs=self.vocab_size + 1,
                        activation_fn=None,
                        scope='logit')
                    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))  # Sample from the distribution
                    self.generator.append(prev_symbol)
                    return tf.nn.embedding_lookup(self.embedding_map, prev_symbol)

            outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_inputs, initial_state, self.lstm,
                                                                        loop_function=loop)
            self.g_output = output = tf.reshape(tf.concat(axis=1, values=outputs[1:]), [-1,
                                                                                        self.embedding_size])  # outputs[1:], because we don't calculate loss on time 0.
            self.g_logits = logits = tf.contrib.layers.fully_connected(
                        inputs=output,
                        num_outputs=self.vocab_size + 1,
                        activation_fn=None,
                        scope='logit')
            self.g_probs = probs = tf.reshape(tf.nn.softmax(logits), [self.batch_size, MAX_STEPS, self.vocab_size + 1])

        self.generator = tf.transpose(tf.reshape(tf.concat(axis=0, values=self.generator), [MAX_STEPS - 1, -1]))

    def summary_saver(self):
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=50)
