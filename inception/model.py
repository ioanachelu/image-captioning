import image_reader
import numpy as np
import tensorflow as tf

from inception import image_embedding

IMG_MEAN = np.array((98, 97, 101), dtype=np.float32)

FLAGS = tf.app.flags.FLAGS


class ImageCaptioningModel():
    def __init__(self, mode='train'):
        self.train_embedding_network = FLAGS.train_embedding_network
        self.weight_initializer = tf.random_uniform_initializer(-FLAGS.weight_initializer_interval,
                                                                FLAGS.weight_initializer_interval)
        self.mode = mode
        self.num_preprocess_threads = 4
        # Reader for the input data.
        self.reader = tf.TFRecordReader()

    def build_image_embeddings(self):
        inception_output = image_embedding.inception_v3(self.images, trainable=self.train_embedding_network,
                                                        is_training=(self.mode == 'train'))
        self.embedding_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

        # Map embedding network output into embedding space.
        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=inception_output,
                num_outputs=FLAGS.embedding_size,
                activation_fn=None,
                weights_initializer=self.weight_initializer,
                biases_initializer=None,
                scope=scope)

        # Save the embedding size in the graph.
        tf.constant(FLAGS.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings

    def build_seq_embeddings(self):
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[FLAGS.vocab_size, FLAGS.embedding_size],
                initializer=self.weight_initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

        self.seq_embeddings = seq_embeddings

    def build_model(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_lstm_units, state_is_tuple=True)
        if self.mode == 'train':
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                                                      output_keep_prob=FLAGS.lstm_dropout_keep_prob)

        with tf.variable_scope("lstm", initializer=self.weight_initializer) as lstm_scope:
            # Feed the image embeddings to set the initial LSTM state.
            zero_state = lstm_cell.zero_state(batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
            _, initial_state = lstm_cell(self.image_embeddings, zero_state)

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()

            if self.mode == 'train' or self.mode == 'test':
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=self.seq_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)
            else:
                # TODO:
                # WHY AM I DOING THIS?????
                # In inference mode, use concatenated states for convenient feeding and
                # fetching.
                tf.concat(initial_state, 1, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                # Run a single LSTM step.
                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(self.seq_embeddings, squeeze_dims=[1]),
                    state=state_tuple)

                # Concatentate the resulting state.
                tf.concat(state_tuple, 1, name="state")

        # Stack batches vertically.
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=FLAGS.vocab_size,
                activation_fn=None,
                weights_initializer=self.weight_initializer,
                scope=logits_scope)

        if self.mode == 'train' or self.mode == 'test':
            targets = tf.reshape(self.target_seqs, [-1])
            weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

            # Compute losses.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)), tf.reduce_sum(weights), name="batch_loss")
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation
        else:
            tf.nn.softmax(logits, name="softmax")

    def load_embedding_network(self):
        if self.mode == 'train' or self.mode == 'test':
            saver = tf.train.Saver(self.embedding_variables)

        def restore_fn(sess):
            tf.logging.info("Restoring Inception variables from checkpoint file %s",
                            FLAGS.embedding_network_checkpoint_file)
            saver.restore(sess, FLAGS.embedding_network_checkpoint_file)

        self.embedding_network_init_fn = restore_fn

    def setup_global_step(self):
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build_input(self):

        if self.mode == 'train' or self.mode == 'test':
            # Prefetch serialized SequenceExample protos.
            input_queue = image_reader.prefetch_input_data(
                self.reader,
                is_training=(self.mode == "train"),
                batch_size=FLAGS.batch_size,
                values_per_tfrec=2300,
                input_queue_capacity_factor=2,
                num_reader_threads=1)


            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion.
            images_and_captions = []
            for thread_id in range(self.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                encoded_image, caption = image_reader.parse_sequence_example(
                    serialized_sequence_example)
                image = image_reader.process_image(encoded_image, (self.mode == 'train'), thread_id=thread_id)
                images_and_captions.append([image, caption])

            queue_capacity = (2 * self.num_preprocess_threads * FLAGS.batch_size)
            images, input_seqs, target_seqs, input_mask = image_reader.batch_with_dynamic_pad(images_and_captions,
                                                                                              batch_size=FLAGS.batch_size,
                                                                                              queue_capacity=queue_capacity)
        else:
            image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[None],  # batch_size
                                        name="input_feed")

            # Process image and insert batch dimensions.
            images = tf.expand_dims(image_reader.process_image(image_feed, False, 0), 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None

        self.images = images
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_mask = input_mask


    def setup(self):
        self.build_input()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.load_embedding_network()
        self.setup_global_step()

    def train(self, sess, g):

        # Set up the learning rate.
        learning_rate_decay_fn = None
        if FLAGS.train_embedding_network:
            learning_rate = tf.constant(FLAGS.train_embedding_network_learning_rate)
        else:
            learning_rate = tf.constant(FLAGS.initial_learning_rate)
            if FLAGS.num_epochs_per_decay > 0:
                num_batches_per_epoch = (FLAGS.num_examples_per_epoch /
                                         FLAGS.batch_size)
                decay_steps = int(num_batches_per_epoch *
                                  FLAGS.num_epochs_per_decay)

                def _learning_rate_decay_fn(learning_rate, global_step):
                    return tf.train.exponential_decay(
                        learning_rate,
                        global_step,
                        decay_steps=decay_steps,
                        decay_rate=FLAGS.learning_rate_decay_factor,
                        staircase=True)

                learning_rate_decay_fn = _learning_rate_decay_fn


        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=self.total_loss,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer='SGD',
            clip_gradients=FLAGS.gradient_clip_value,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=5)

        # Run training.
        tf.contrib.slim.learning.train(
            train_op,
            FLAGS.checkpoint_dir,
            log_every_n_steps=FLAGS.checkpoint_every,
            graph=g,
            global_step=self.global_step,
            number_of_steps=FLAGS.num_steps,
            init_fn=self.embedding_network_init_fn,
            saver=saver)


    def train2(self, sess, g):

        # # Set up the learning rate.
        # learning_rate_decay_fn = None
        # if FLAGS.train_embedding_network:
        #     learning_rate = tf.constant(FLAGS.train_embedding_network_learning_rate)
        # else:
        #     learning_rate = tf.constant(FLAGS.initial_learning_rate)
        #     if FLAGS.num_epochs_per_decay > 0:
        #         num_batches_per_epoch = (FLAGS.num_examples_per_epoch /
        #                                  FLAGS.batch_size)
        #         decay_steps = int(num_batches_per_epoch *
        #                           FLAGS.num_epochs_per_decay)
        #
        #         def _learning_rate_decay_fn(learning_rate, global_step):
        #             return tf.train.exponential_decay(
        #                 learning_rate,
        #                 global_step,
        #                 decay_steps=decay_steps,
        #                 decay_rate=FLAGS.learning_rate_decay_factor,
        #                 staircase=True)
        #
        #         learning_rate_decay_fn = _learning_rate_decay_fn
        num_batches_per_epoch = (FLAGS.num_examples_per_epoch / FLAGS.batch_size)

        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(
                        FLAGS.initial_learning_rate,
                        self.global_step,
                        decay_steps=decay_steps,
                        decay_rate=FLAGS.learning_rate_decay_factor,
                        staircase=True)

        opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]

        grads = tf.gradients(self.total_loss, all_trainable)
        grads, _ = tf.clip_by_global_norm(grads, FLAGS.gradient_clip_value)
        # for grad, weight in zip(grads, all_trainable):
        #     total_summary.append(tf.summary.histogram(weight.name + '_grad', grad))
        #     total_summary.append(tf.summary.histogram(weight.name, weight))

        train_op = opt.apply_gradients(zip(grads, all_trainable))

        increment_global_step = self.global_step.assign_add(1)

        sess.run(self.embedding_network_init_fn)

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=restore_var, max_to_keep=1000)
        # Iterate over training steps.
        step_count = sess.run(self.global_step)

        while True:
            if step_count == FLAGS.num_steps:
                break
            start_time = time.time()

            if step_count % FLAGS.checkpoint_every == 0:
                loss_value, mae_pred, step_mae_pred, step_avg_mae_pred, _, images, labels, preds, summary_images, summary, _, _ = sess.run(
                    [reduced_loss, mae, step_mae, step_avg_mae, update_op, image_batch, label_batch, logits,
                     total_image_summary,
                     merged_summary, train_op, increment_global_step])
                summary_writer.add_summary(summary_images, step_count)
                summary_writer.add_summary(summary, step_count)
                self.save_model(saver, sess, FLAGS.checkpoint_dir, step_count, self.global_step)
            elif step_count % FLAGS.summary_every == 0:
                loss_value, mae_pred, step_mae_pred, step_avg_mae_pred, _, images, labels, preds, summary_images, summary, _, _ = sess.run(
                    [reduced_loss, mae, step_mae, step_avg_mae, update_op, image_batch, label_batch, logits,
                     total_image_summary,
                     merged_summary, train_op, increment_global_step])
                summary_writer.add_summary(summary_images, step_count)
                summary_writer.add_summary(summary, step_count)
            else:
                loss_value, mae_pred, step_mae_pred, step_avg_mae_pred, _, _, _ = sess.run(
                    [reduced_loss, mae, step_mae, step_avg_mae, update_op, train_op, increment_global_step])
            duration = time.time() - start_time
            # if step_count % FLAGS.summary_every == 0:
            print(
                'step {:d} \t loss = {:.3f}, step_mae = {:.3f}, mae_exp_avg = {:.3f}, running_avg_mae = {:.3f}, ({:.3f} sec/step)'.format(
                    step_count, loss_value, step_mae_pred, step_avg_mae_pred, mae_pred,
                    duration))
            step_count += 1

        # # Set up the training ops.
        # train_op = tf.contrib.layers.optimize_loss(
        #     loss=self.total_loss,
        #     global_step=self.global_step,
        #     learning_rate=learning_rate,
        #     optimizer='SGD',
        #     clip_gradients=FLAGS.gradient_clip_value,
        #     learning_rate_decay_fn=learning_rate_decay_fn)
        #
        # # Set up the Saver for saving and restoring model checkpoints.
        # saver = tf.train.Saver(max_to_keep=5)
        #
        # # Run training.
        # tf.contrib.slim.learning.train(
        #     train_op,
        #     FLAGS.checkpoint_dir,
        #     log_every_n_steps=FLAGS.checkpoint_every,
        #     graph=g,
        #     global_step=self.global_step,
        #     number_of_steps=FLAGS.num_steps,
        #     init_fn=self.embedding_network_init_fn,
        #     saver=saver)



