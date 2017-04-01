import numpy as np
import tensorflow as tf
import utils
import vgg

FLAGS = tf.app.flags.FLAGS
# max number of words in caption at test time
MAX_STEPS = 30


# Caption generator model
class ShowAndTell():
    def __init__(self, vocab_size, seq_length, sess):
        self.embedding_size = np.int(FLAGS.embedding_size)
        self.num_lstm_units = np.int(FLAGS.embedding_size)
        self.batch_size = np.int(FLAGS.batch_size)
        self.seq_length = np.int(seq_length)
        self.vocab_size = np.int(vocab_size)
        self.sess = sess

        # Variable indicating in training mode or evaluation mode - for dropout keep probability
        self.training = tf.Variable(True, trainable=False, name="training")

        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], name="images")
        self.captions = tf.placeholder(tf.int32, [None, self.seq_length + 2])
        self.masks = tf.placeholder(tf.float32, [None, self.seq_length + 2])

        # used to initialize the weights of the logits output
        self.initializer = tf.random_uniform_initializer(
            minval=-FLAGS.initializer_scale,
            maxval=FLAGS.initializer_scale)

        self.keep_prob = tf.cond(self.training,
                                 lambda: tf.constant(1 - FLAGS.lstm_dropout_keep_prob),
                                 lambda: tf.constant(1.0), name='keep_prob')

        # if resuming to not use pretrained weights because weights are already saved in the model
        if FLAGS.resume:
            pretrained_weights = None
        else:
            pretrained_weights = FLAGS.pretrained_weights

        # load VGG with pretrained weights
        self.cnn = vgg.Vgg16(pretrained_weights)
        with tf.variable_scope("cnn") as scope:
            self.cnn.build(self.images)
        # recover the last fully connected layer before the top layer in classification
        self.fc7 = self.cnn.drop7
        self.cnn_training = self.cnn.training

        with tf.variable_scope("lstm"):
            # create sequence embedding mapping from the vocabulary + EOS to an embedding size of 512
            with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
                self.embedding_map = tf.get_variable(
                    name="map",
                    shape=[self.vocab_size + 1, self.embedding_size],
                    initializer=self.initializer)

            # Create LSTM cell
            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.num_lstm_units)
            # Add dropout on the output layer
            self.lstm = tf.contrib.rnn.DropoutWrapper(self.lstm, 1.0, self.keep_prob)

        # Transform the last fully connected layer of the CNN into a fixed embedding vector of 512 units
        # to pass to the RNN
        with tf.variable_scope("cnn"):
            self.image_emb = tf.contrib.layers.fully_connected(
                inputs=self.fc7,
                num_outputs=self.embedding_size,
                activation_fn=None,
                scope='encode_image')

    # Build model for training step.
    def build_model(self):
        with tf.name_scope("batch_size"):
            # Get batch_size from the first dimension of self.images
            self.batch_size = tf.shape(self.images)[0]

        # ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, name='loss_avg')

        with tf.variable_scope("lstm") as lstm_scope:
            # Replicate self.seq_per_img times for each image embedding because captions and
            #  masks come with batchsize * 5
            image_emb = tf.reshape(tf.tile(tf.expand_dims(self.image_emb, 1), [1, 5, 1]),
                                   [self.batch_size * 5, self.embedding_size])

            # Get embedding for current caption batch trimmed at sequence length + EOS token
            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.embedding_map, self.captions[:, :self.seq_length + 1])

            # Split the sequence from tensor to list of tensors and get rid of shape 1 from axis 1
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.seq_length + 1, value=current_emb)
            rnn_inputs = [tf.squeeze(input_, [1]) for input_ in rnn_inputs]
            # Concatenate the caption inputs with the first input containing the image
            rnn_inputs = [image_emb] + rnn_inputs

            # The initial state is zero. Batch size is multiplied by 5 because captions and
            #  masks come with batchsize * 5
            initial_state = self.lstm.zero_state(self.batch_size * 5, tf.float32)

            # Run the LSTM for seq_length steps and get the outputs
            outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_inputs, initial_state, self.lstm,
                                                                        loop_function=None)
            # outputs, last_state = tf.nn.dynamic_rnn(self.lstm, rnn_inputs, initial_state=initial_state)

            # Stack all the outputs from the outputs list
            outputs = tf.concat(axis=0, values=outputs[1:])
            # Transform them into logits with the vocabulary size plus the EOS token
            self.logits_flat = tf.contrib.layers.fully_connected(
                inputs=outputs,
                num_outputs=self.vocab_size + 1,
                activation_fn=None,
                scope='logit')
            # Split them into a sequence again to output a list of tensors
            self.logits = tf.split(axis=0, num_or_size_splits=len(rnn_inputs) - 1, value=self.outputs)


        with tf.variable_scope("loss"):
            # Compute the cross entropy loss with logits and sparse target labels weighted by the masks
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(self.logits,
                                                                      [tf.squeeze(label, [1]) for label in
                                                                       tf.split(axis=1,
                                                                                num_or_size_splits=self.seq_length + 1,
                                                                                value=self.captions[:, 1:])],
                                                                      [tf.squeeze(mask, [1]) for mask in
                                                                       tf.split(axis=1,
                                                                                num_or_size_splits=self.seq_length + 1,
                                                                                value=self.masks[:, 1:])])
            # Sum all the losses into a total_loss
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
        rnn_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')
        # clip the gradients by value. Should try by global norm also.
        rnn_grads = utils.clip_by_value(tf.gradients(self.loss, rnn_weights), -FLAGS.gradient_clip_value,
                                        FLAGS.gradient_clip_value)

        # Add summaries for each weight and gradient
        for grad, weight in zip(rnn_grads, rnn_weights):
            tf.summary.histogram(weight.name + '_grad', grad)
            tf.summary.histogram(weight.name, weight)

        # Apply gradients for RNN weights
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.8, beta2=0.999, epsilon=1e-8)
        self.train_op = optimizer.apply_gradients(zip(rnn_grads, rnn_weights))

        # Collect the cnn variables, and create the optimizer of cnn
        cnn_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
        cnn_grads = utils.clip_by_value(tf.gradients(self.loss, cnn_weights), -FLAGS.gradient_clip_value,
                                        FLAGS.gradient_clip_value)
        # Add summaries for each weight and gradient
        for grad, weight in zip(cnn_grads, cnn_weights):
            if grad is not None and weight is not None:
                tf.summary.histogram(weight.name + '_grad', grad)
                tf.summary.histogram(weight.name, weight)

        # Apply gradients for CNN
        cnn_optimizer = tf.train.AdamOptimizer(self.cnn_lr, beta1=0.8, beta2=0.999, epsilon=1e-8)
        self.cnn_train_op = cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_weights))

        # Add summaries for training loss and learning rates
        tf.summary.scalar('training loss', self.loss)
        tf.summary.scalar('learning rate', self.lr)
        tf.summary.scalar('cnn learning rate', self.cnn_lr)
        self.summaries = tf.summary.merge_all()


    # Build model for evaluation procedure
    def build_generator(self):
        # Generated sequence
        self.generator = []
        with tf.variable_scope("lstm") as lstm_scope:
            # Pas zero as BOS token
            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.embedding_map, tf.zeros([self.batch_size], tf.int32))

            # Construct input sequence with the image embedding the BOS
            rnn_inputs = [self.image_emb] + [current_emb] + [0] * (MAX_STEPS - 1)

            initial_state = self.lstm.zero_state(self.batch_size, tf.float32)

            tf.get_variable_scope().reuse_variables()

            def decode_greedy(prev, i):
                if i == 1:
                    # if first output, pass the first input
                    return rnn_inputs[1]
                # else sample using the previous output passed through the fc to get logit
                with tf.variable_scope(lstm_scope):
                    prev = tf.contrib.layers.fully_connected(
                        inputs=prev,
                        num_outputs=self.vocab_size + 1,
                        activation_fn=None,
                        scope='logit')
                    # Stop gradient flow. We are just sampling the max value
                    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                    # add decoded symbol to the generated sequence
                    self.generator.append(prev_symbol)
                    # encode the max symbol and pass it on to the rnn as input
                    return tf.nn.embedding_lookup(self.embedding_map, prev_symbol)

            outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(rnn_inputs, initial_state, self.lstm,
                                                                        loop_function=decode_greedy)

            # Start the loss at the second output
            self.g_outputs = tf.reshape(tf.concat(axis=1, values=outputs[1:]), [-1, self.embedding_size])
            self.g_logits = tf.contrib.layers.fully_connected(
                        inputs=self.g_outputs,
                        num_outputs=self.vocab_size + 1,
                        activation_fn=None,
                        scope='logit')
            self.g_probs = tf.reshape(tf.nn.softmax(self.g_logits), [self.batch_size, MAX_STEPS, self.vocab_size + 1])

        self.generator = tf.transpose(tf.reshape(tf.concat(axis=0, values=self.generator), [MAX_STEPS - 1, -1]))

    def summary_saver(self):
        self.summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=50)
