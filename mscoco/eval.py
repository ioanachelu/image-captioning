import json
import numpy as np
import tensorflow as tf
import time
import os
from model import ShowAndTell
from dataloader import DataLoader
import eval_utils
import flags
import h5py
import utils
FLAGS = tf.app.flags.FLAGS
NUM_THREADS = 2

def eval_split(sess, model, loader, eval_kwargs):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('split', 'test')
    language_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    sess.run(tf.assign(model.training, False))
    sess.run(tf.assign(model.cnn_training, False))

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        # fetch a batch of data
        if opt.beam_size > 1:
            data = loader.get_batch(split, 1)
            n = n + 1
        else:
            data = loader.get_batch(split, opt.batch_size)
            n = n + opt.batch_size

        #evaluate loss if we have the labels
        loss = 0
        if data.get('labels', None) is not None:
            # forward the model to get loss
            feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
            loss = sess.run(model.cost, feed)
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        if opt.beam_size == 1:
            # forward the model to also get generated samples for each image
            feed = {model.images: data['images']}
            #g_o,g_l,g_p, seq = sess.run([model.g_output, model.g_logits, model.g_probs, model.generator], feed)
            seq = sess.run(model.generator, feed)

            #set_trace()
            sents = utils.decode_sequence(vocab, seq)

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
                if verbose:
                    print('image %s: %s' %(entry['image_id'], entry['caption']))
        else:
            seq = model.decode(data['images'], opt.beam_size, sess)
            sents = [' '.join([vocab.get(str(ix), '') for ix in sent]).strip() for sent in seq]
            sents = [sents[0]]
            entry = {'image_id': data['infos'][0]['id'], 'caption': sents[0]}
            predictions.append(entry)
            if verbose:
                for sent in sents:
                    print('image %s: %s' %(entry['image_id'], sent))

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if opt.dump_path == 1:
                entry['file_name'] = data['infos'][k]['file_path']
                table.insert(predictions, entry)
            if opt.dump_images == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(opt.image_root, data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if language_eval == 1:
        lang_stats = eval_utils.language_eval(dataset, predictions)

    # Switch back to training mode
    sess.run(tf.assign(model.training, True))
    sess.run(tf.assign(model.cnn_training, True))
    return loss_sum/loss_evals, predictions, lang_stats
# print('Loading json file: ', FLAGS.output_json)
# info = json.load(open(FLAGS.output_json))
# vocab = info['index_to_word']
# vocab_size = len(vocab)
# seq_per_img = 5
# print('vocab size is ', vocab_size)
#
# # open the hdf5 file
# print('Loading h5 file: ', FLAGS.output_h5)
# h5_file = h5py.File(FLAGS.output_h5)

# Setup the model
def eval():
    loader = DataLoader()
    vocab_size = loader.vocab_size
    seq_length = loader.seq_length

    # Evaluation fun(ction)


    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        model = ShowAndTell(vocab_size, seq_length, sess)
        model.build_model()
        model.build_generator()

        # Initilize the variables
        sess.run(tf.global_variables_initializer())

        # Load the model checkpoint to evaluate
        utils.load_model(model.saver, sess)
        sess.run(tf.local_variables_initializer())

        # Set sample options
        eval_kwargs = {'test_images_use': FLAGS.test_images_use,
                       'split': 'test',
                       'language_eval': FLAGS.language_eval,
                       'dataset': FLAGS.output_json}

        loss, predictions, lang_stats = eval_utils.test_eval(sess, model, loader, eval_kwargs)

    print('loss: ', loss)
    if lang_stats:
      print(lang_stats)


    json.dump(predictions, open('data/predictions.json', 'w'))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    eval()