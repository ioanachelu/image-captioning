import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

class ShowAndTell():
    def __init__(self, image_embedding_size, embedding_size, num_lstm_units, batch_size, n_lstm_steps, n_words, bias_init_vector=None, mode="train"):

        self.image_embedding_size = np.int(image_embedding_size)
        self.embedding_size = np.int(embedding_size)
        self.num_lstm_units = np.int(num_lstm_units)
        self.batch_size = np.int(batch_size)
        self.n_lstm_steps = np.int(n_lstm_steps)
        self.n_words = np.int(n_words)
        self.mode = mode
        self.initializer = tf.random_uniform_initializer(
            minval=-FLAGS.initializer_scale,
            maxval=FLAGS.initializer_scale)

        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            self.embedding_map = tf.get_variable(
                name="map",
                shape=[self.n_words, self.embedding_size],
                initializer=self.initializer)

        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.num_lstm_units)
        if self.mode == "train":
            self.lstm = tf.contrib.rnn.DropoutWrapper(
                self.lstm,
                input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                output_keep_prob=FLAGS.lstm_dropout_keep_prob)

        self.embed_word_W = tf.get_variable(shape=[num_lstm_units, n_words], initializer=self.initializer, name='embed_word_W')

        # if bias_init_vector is not None:
        #     self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        # else:
        #     self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.image_embedding_size])
        sentence = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        # image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b  # (batch_size, dim_hidden)

        with tf.variable_scope("image_embedding") as scope:
            image_emb = tf.contrib.layers.fully_connected(
                inputs=image,
                num_outputs=self.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)

        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, name='loss_avg')
        generated_words = [tf.cast(sentence[:, 0], dtype=tf.int64)]

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            zero_state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            _, state = self.lstm(image_emb, zero_state)

            with tf.device("/cpu:0"):
                current_emb = tf.nn.embedding_lookup(self.embedding_map, sentence)

            lstm_scope.reuse_variables()


            for i in range(self.n_lstm_steps - 1):
                # print(i)
                output, state = self.lstm(current_emb[:, i], state)

                labels = tf.expand_dims(sentence[:, i + 1], 1)  # (batch_size)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat([indices, labels], 1)
                onehot_labels = tf.sparse_to_dense(
                    concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)  # (batch_size, n_words)

                logit_words = tf.matmul(output, self.embed_word_W)# + self.embed_word_b  # (batch_size, n_words)
                batch_loss_per_word = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                batch_loss_per_word = batch_loss_per_word * mask[:, i + 1]  # tf.expand_dims(mask, 1)

                batch_loss_per_word = tf.reduce_sum(batch_loss_per_word)
                tf.losses.add_loss(batch_loss_per_word)
                # Add summaries.
                # tf.summary.scalar("losses/batch_loss_raw", batch_loss_per_word)

                max_prob_word = tf.argmax(logit_words, 1)
                generated_words.append(max_prob_word)

        total_loss = tf.losses.get_total_loss()
        total_loss = total_loss / tf.reduce_sum(mask[:, 1:])
        tf.summary.scalar("losses/total_loss", total_loss)
        self.maintain_averages_op = ema.apply([total_loss])
        tf.summary.scalar("losses/avg_total_loss", ema.average(total_loss))

        # for var in tf.trainable_variables():
        #     tf.summary.histogram("parameters/" + var.op.name, var)

        self.loss = total_loss
        generated_words = tf.stack(generated_words, axis=1)
        # generated_words = tf.boolean_mask(generated_words, tf.cast(mask[:, 1:], dtype=tf.bool))
        return total_loss, image, sentence, mask, generated_words

    def train(self, global_step, num_examples_per_epoch):
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        decay_steps = int(num_examples_per_epoch * FLAGS.num_epochs_per_decay)

        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step,
                                                   decay_steps, FLAGS.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('LR', learning_rate)
        with tf.control_dependencies([self.maintain_averages_op]):
            # opt = tf.train.AdamOptimizer(learning_rate)
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            grads = tf.gradients(self.loss, all_trainable)

            for grad, weight in zip(grads, all_trainable):
                tf.summary.histogram(weight.name + '_grad', grad)
                tf.summary.histogram(weight.name, weight)

            grads, grad_norms = tf.clip_by_global_norm(grads, FLAGS.gradient_clip_value)
            train_op = opt.apply_gradients(zip(grads, all_trainable))


        return train_op

    def summary(self):
        summary_writer = tf.summary.FileWriter(FLAGS.summaries_dir, graph=tf.get_default_graph())
        merged_summary = tf.summary.merge_all()
        return summary_writer, merged_summary

    def build_generator(self, maxlen):
        word_to_index = np.load("data/word_to_index.npy")[()]
        end_word_tensor = tf.expand_dims(tf.constant(word_to_index[FLAGS.end_word], dtype=tf.int64), 0)
        image = tf.placeholder(tf.float32, [1, self.image_embedding_size])
        with tf.variable_scope("image_embedding") as scope:
            image_emb = tf.contrib.layers.fully_connected(
                inputs=image,
                num_outputs=self.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                scope=scope)
        generated_words = []
        mask = []
        with tf.variable_scope("lstm") as lstm_scope:
            zero_state = self.lstm.zero_state(batch_size=1, dtype=tf.float32)
            _, state = self.lstm(image_emb, zero_state)
            with tf.device("/cpu:0"):

                last_word = tf.nn.embedding_lookup(self.embedding_map, [word_to_index[FLAGS.start_word]])

            tf.get_variable_scope().reuse_variables()

            for i in range(maxlen):
                output, state = self.lstm(last_word, state)
                logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                max_prob_word = tf.argmax(logit_words, 1)
                with tf.device("/cpu:0"):
                    last_word = tf.nn.embedding_lookup(self.Wemb, max_prob_word)
                last_word += self.bemb

                mask.append(tf.equal(max_prob_word, end_word_tensor))
                generated_words.append(max_prob_word)

        return image, generated_words, mask