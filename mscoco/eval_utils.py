import json
from json import encoder
import utils
import tensorflow as tf
import os


def language_eval(dataset, preds):
    import sys
    if 'coco' in dataset:
        annFile = 'annotations/captions_val2014.json'
    else:
        sys.path.append("data")
        annFile = 'data/flikr.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    coco = COCO(annFile)
    valids = coco.getImgIds()
    # valids = [int(v) for v in valids]
    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open('tmp.json', 'w')) # serialize to temporary json file. Sigh, COCO API...

    resFile = 'tmp.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out


def val_eval(sess, model, loader, eval_kwargs):
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    language_eval = eval_kwargs.get('language_eval', True)
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    sess.run(tf.assign(model.training, False))
    sess.run(tf.assign(model.cnn_training, False))

    loader.reset_iterator(split)
    # maxlen = 30

    predictions = []
    loss_sum = 0
    loss_evals = 0
    step = 0

    while True:
        data = loader.get_batch(split, 1)
        step += 1

        # forward the model to get loss
        feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
        loss = sess.run(model.loss, feed)
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        feed = {model.images: data['images']}
        seq = sess.run(model.generator, feed)

        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(step - ix1):
            predictions.pop()

        print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if step >= val_images_use:
            break

    if language_eval:
        lang_stats = language_eval(dataset, predictions)

    # Switch back to training mode
    sess.run(tf.assign(model.training, True))
    sess.run(tf.assign(model.cnn_training, True))
    return loss_sum / loss_evals, predictions, lang_stats


def test_eval(sess, model, loader, eval_kwargs):
    test_images_use = eval_kwargs.get('test_images_use', -1)
    split = eval_kwargs.get('split', 'test')
    language_eval = eval_kwargs.get('language_eval', True)
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    sess.run(tf.assign(model.training, False))
    sess.run(tf.assign(model.cnn_training, False))

    loader.reset_iterator(split)
    # maxlen = 30

    predictions = []
    loss_sum = 0
    loss_evals = 0
    step = 0

    while True:
        data = loader.get_batch(split, 1)
        step += 1

        # forward the model to get loss
        feed = {model.images: data['images'], model.labels: data['labels'], model.masks: data['masks']}
        loss = sess.run(model.loss, feed)
        loss_sum = loss_sum + loss
        loss_evals += 1

        feed = {model.images: data['images']}
        seq = sess.run(model.generator, feed)

        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)

            cmd = 'cp "' + data['infos'][k]['file_path'] + '" captions/img' + str(data['infos'][k]['id']) + '.jpg'  # bit gross
            print(cmd)
            os.system(cmd)

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

    if language_eval:
        lang_stats = language_eval(dataset, predictions)


    # Switch back to training mode
    sess.run(tf.assign(model.training, True))
    sess.run(tf.assign(model.cnn_training, True))
    return loss_sum / loss_evals, predictions, lang_stats
