import json
from json import encoder
import utils
import tensorflow as tf
import os
import flags
FLAGS = tf.app.flags.FLAGS

# Evaluate predictions using NLP metrics: BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE and CIDEr. METEOR is skiped because the
# coco-caption code has some leaks and so as not to mess up the run
def language_eval(dataset_json, preds):
    import sys
    # for MSCOCO use the official captions for validation
    if 'coco' in dataset_json:
        annFile = 'annotations/captions_val2014.json'
    # for flikr use the one created by me. Not used at this moment
    else:
        sys.path.append("data")
        annFile = 'data/flikr.json'

    # borrowed sources from coco-caption project
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    # use json as per coco-caption implementation
    json.dump(preds_filt, open('tmp.json', 'w'))

    resFile = 'tmp.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary using NLP metrics results
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out


def greey_sampling(model, loader, data, predictions, sess):
    # do an inverence to get the predicted caption sequences
    feed = {model.images: data['images']}
    seq = sess.run(model.generator, feed)

    # decode sequences to words from indexes
    sents = utils.decode_sequence(loader.get_vocab(), seq)

    #load predictions in a dictionary with image id and predicted captions to be used by the language eval function
    for k, sent in enumerate(sents):
        entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
        predictions.append(entry)


def beam_search(model, loader, data, predictions, sess):
    seq = model.decode(data['images'], sess)
    sents = [' '.join([loader.get_vocab().get(str(index), '') for index in sent]).strip() for sent in seq]
    entry = {'image_id': data['infos'][0]['id'], 'caption': sents[0]}
    predictions.append(entry)


# Do one step of evaluation using the validation dataset
def val_eval(sess, model, loader, eval_kwargs):
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    language_eval = eval_kwargs.get('language_eval', True)
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    sess.run(tf.assign(model.training, False))
    sess.run(tf.assign(model.cnn_training, False))

    loader.reset_iterator(split)

    predictions = []
    loss_sum = 0
    loss_evals = 0
    step = 0

    while True:
        # Evaluate with batch size 1
        data = loader.get_batch(split, 1)
        step += 1

        # forward the model to get loss
        feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
        loss = sess.run(model.loss, feed)

        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        if FLAGS.beam_search_size == 1:
            greey_sampling(model, loader, data, predictions, sess)
        else:
            beam_search(model, loader, data, predictions, sess)

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        # if processed enough images or all the images in an epoch -> stop
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(step - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if step >= val_images_use:
            break

    # Evaluate predictions using coco-caption to get NLP metrics results
    if language_eval:
        lang_stats = language_eval(dataset, predictions)

    # Switch back to training mode
    sess.run(tf.assign(model.training, True))
    sess.run(tf.assign(model.cnn_training, True))

    return loss_sum / loss_evals, predictions, lang_stats


# Do evaluation using test data
def test_eval(sess, model, loader, eval_kwargs):
    test_images_use = eval_kwargs.get('test_images_use', -1)
    split = eval_kwargs.get('split', 'test')
    language_eval = eval_kwargs.get('language_eval', True)
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode - for dropout behaviour
    sess.run(tf.assign(model.training, False))
    sess.run(tf.assign(model.cnn_training, False))

    loader.reset_iterator(split)

    predictions = []
    loss_sum = 0
    loss_evals = 0
    step = 0

    while True:
        # use batch size 1
        data = loader.get_batch(split, 1)
        step += 1

        # forward the model to get loss
        feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
        loss = sess.run(model.loss, feed)

        loss_sum = loss_sum + loss
        loss_evals += 1

        # forward to get predicted caption sequences
        feed = {model.images: data['images']}
        seq = sess.run(model.generator, feed)

        # transform caption indices to words from vocabulary
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        # create prediction dictionary to use with coco-caption
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)

            # copy images to captions folder using their id as name to be later retrieved using the resulted caption json
            cmd = 'cp "' + data['infos'][k]['file_path'] + '" captions/img' + str(data['infos'][k]['id']) + '.jpg'  # bit gross
            print(cmd)
            os.system(cmd)

        # stoped if processed all test images
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if test_images_use != -1:
            ix1 = min(ix1, test_images_use)
        for i in range(step - ix1):
            predictions.pop()

        if data['bounds']['wrapped']:
            break
        if step >= test_images_use:
            break

    # Evaluate NLP metrics
    if language_eval:
        lang_stats = language_eval(dataset, predictions)


    # Switch back to training mode
    sess.run(tf.assign(model.training, True))
    sess.run(tf.assign(model.cnn_training, True))
    return loss_sum / loss_evals, predictions, lang_stats
