import tensorflow as tf
import flags
FLAGS = tf.app.flags.FLAGS


# Clip each value in the gradient tensor between the clip_value_min and clip_value_max as recommended in Karpathy
# Deep Visual-Semantic Alignments for Generating Image Descriptions
def clip_by_value(tensor_list, clip_value_min, clip_value_max, name=None):
    tensor_list = list(tensor_list)

    with tf.name_scope(name or "clip_by_value") as name:
        values = [
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i)
            if t is not None else t
            for i, t in enumerate(tensor_list)]
        values_clipped = []
        for i, v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.get_default_graph().colocate_with(v):
                    values_clipped.append(
                        tf.clip_by_value(v, clip_value_min, clip_value_max))

        list_clipped = [
            tf.IndexedSlices(c_v, t.indices, t.dense_shape)
            if isinstance(t, tf.IndexedSlices)
            else c_v
            for (c_v, t) in zip(values_clipped, tensor_list)]

    return list_clipped


# Transform output_sequence from word indexes to words - index 0 is reserved for EOS
def decode_sequence(index_to_word, output_sequence):
    batch, caption_length = output_sequence.shape
    out = []
    for i in range(batch):
        caption = ' '.join([index_to_word[str(output_sequence[i, j])] for j in range(caption_length) if output_sequence[i, j] > 0])
        out.append(caption)
    return out


# Load model from a previous checkpoint
def load_model(saver, sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restored model parameters from {}".format(ckpt.model_checkpoint_path))

