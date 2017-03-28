import numpy as np
import tensorflow as tf
from utils import clip_by_value

FLAGS = tf.app.flags.FLAGS

class ShowAndTell():
    def __init__(self, vocab_size, seq_length):
        self.embedding_size = np.int(FLAGS.embedding_size)
        self.num_lstm_units = np.int(FLAGS.embedding_size)
        self.batch_size = np.int(FLAGS.batch_size)
        self.seq_length = np.int(seq_length)
        self.vocab_size = np.int(vocab_size)

        # Variable indicating in training mode or evaluation mode
        self.training = tf.Variable(True, trainable=False, name="training")

        self.images = tf.placeholder(tf.float32, [self.batch_size, 224, 224, 3], name="images")
        self.labels = tf.placeholder(tf.int32, [self.batch_size, self.seq_length + 2])
        self.masks = tf.placeholder(tf.float32, [self.batch_size, self.seq_length + 2])

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

            self.embed_word_W = tf.get_variable(shape=[num_lstm_units, n_words], initializer=self.initializer,
                                                name='embed_word_W')

        with tf.variable_scope("cnn") as scope:
            self.image_emb = tf.contrib.layers.fully_connected(
                inputs=self.fc7,
                num_outputs=self.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

    def build_model(self):
        with tf.name_scope("batch_size"):
            # Get batch_size from the first dimension of self.images
            self.batch_size = tf.shape(self.images)[0]

        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, name='loss_avg')
        # self.generated_words = [tf.cast(self.labels[:, 0], dtype=tf.int64)]

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            zero_state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            _, state = self.lstm(self.image_emb, zero_state)

            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.embedding_map, self.sentence)

            lstm_scope.reuse_variables()

            for i in range(self.n_lstm_steps - 1):
                # print(i)
                output, state = self.lstm(current_emb[:, i], state)

                labels = tf.expand_dims(self.sentence[:, i + 1], 1)  # (batch_size)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat([indices, labels], 1)
                onehot_labels = tf.sparse_to_dense(
                    concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)  # (batch_size, n_words)

                logit_words = tf.matmul(output, self.embed_word_W)  # + self.embed_word_b  # (batch_size, n_words)
                batch_loss_per_word = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                batch_loss_per_word = batch_loss_per_word * self.mask[:, i + 1]  # tf.expand_dims(mask, 1)

                batch_loss_per_word = tf.reduce_sum(batch_loss_per_word)
                tf.losses.add_loss(batch_loss_per_word)
                # Add summaries.
                # tf.summary.scalar("losses/batch_loss_raw", batch_loss_per_word)

                max_prob_word = tf.argmax(logit_words, 1)
                self.generated_words.append(max_prob_word)

        self.total_loss = tf.losses.get_total_loss()
        self.total_loss = self.total_loss / tf.reduce_sum(self.mask[:, 1:])
        tf.summary.scalar("losses/total_loss", self.total_loss)
        self.maintain_averages_op = ema.apply([self.total_loss])
        tf.summary.scalar("losses/avg_total_loss", ema.average(self.total_loss))

        # for var in tf.trainable_variables():
        #     tf.summary.histogram("parameters/" + var.op.name, var)

        self.generated_words = tf.stack(self.generated_words, axis=1)
        # generated_words = tf.boolean_mask(generated_words, tf.cast(mask[:, 1:], dtype=tf.bool))
        # return self.total_loss, self.image, self.sentence, self.mask, self.generated_words

    def train(self, global_step, num_examples_per_epoch):
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        if FLAGS.num_epochs_per_decay != 0:
            decay_steps = int(num_examples_per_epoch * FLAGS.num_epochs_per_decay)

            learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step,
                                                       decay_steps, FLAGS.learning_rate_decay_factor, staircase=True)
        else:
            learning_rate = FLAGS.initial_learning_rate

        tf.summary.scalar('LR', learning_rate)

        with tf.control_dependencies([self.maintain_averages_op]):
            # opt = tf.train.AdamOptimizer(learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-8)
            # opt = tf.train.RMSPropOptimizer(learning_rate, FLAGS.momentum)

            # Collect the rnn variables, and create the optimizer of rnn
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lstm')
            grads = clip_by_value(tf.gradients(self.total_loss, tvars), -FLAGS.grad_clip, FLAGS.grad_clip)

            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-8)
            train_op = opt.apply_gradients(zip(grads, tvars))
            for grad, weight in zip(grads, tvars):
                tf.summary.histogram(weight.name + '_grad', grad)
                tf.summary.histogram(weight.name, weight)

            cnn_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
            cnn_grads = utils.clip_by_value(tf.gradients(self.cost, cnn_tvars), -self.opt.grad_clip,
                                            self.opt.grad_clip)
            # cnn_grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, cnn_tvars),
            #        self.opt.grad_clip)
            cnn_optimizer = tf.train.AdamOptimizer(FLAGS.cnn_lr, beta1=0.8, beta2=0.999, epsilon=1e-8)
            cnn_train_op = cnn_optimizer.apply_gradients(zip(cnn_grads, cnn_tvars))

            # grads = tf.gradients(self.total_loss, all_trainable)
            # grads = clip_by_value(grads, -FLAGS.gradient_clip_value, FLAGS.gradient_clip_value)

            # grads, grad_norms = tf.clip_by_global_norm(grads, FLAGS.gradient_clip_value)
            # train_op = opt.apply_gradients(zip(grads, all_trainable))

        return train_op, cnn_train_op

    def summary(self):
        summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph=tf.get_default_graph())
        merged_summary = tf.summary.merge_all()
        return summary_writer, merged_summary