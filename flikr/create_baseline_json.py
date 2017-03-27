import json
import codecs
from random import randint
from utils import get_caption_data, preprocess_for_test
import flags
import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS

TOTAL_CAPTIONS_FOR_AN_IMAGE = 5  # total captions for an image in the dataset

output_path = './data/'

feats, captions, filenames_to_captions = get_caption_data(mode=FLAGS.validate_on)


# lets give fixed values to mandatory fields
license_ = 3
url_ = 'asdasdsda.com'
width_ = 640
height_ = 480
date_captured = 14

out_json_tr = []
captions_tr = []
ims = []
anns = []
offset = 0
found = 0
id_ = 0
index_ = 0

filenames_captions_dict = {}
for f, c in filenames_to_captions:
    filenames_captions_dict.setdefault(f, [])
    filenames_captions_dict[f].append(c)

if not tf.gfile.Exists('data/' + FLAGS.validate_on + '_image_id_to_filename.npy') or not tf.gfile.Exists('data/' + FLAGS.validate_on + '_filename_to_image_id.npy'):
    print("Recreating filename image id associations")
    image_id_to_filename = []
    for f in filenames_captions_dict.keys():
        image_id_to_filename.append(f)

    filename_to_image_id = {}
    for id, f in enumerate(image_id_to_filename):
        filename_to_image_id[f] = id

    np.save('data/' + FLAGS.validate_on + '_image_id_to_filename.npy', image_id_to_filename)
    np.save('data/' + FLAGS.validate_on + '_filename_to_image_id.npy', filename_to_image_id)
else:
    print("Loading filename image id associations")
    image_id_to_filename = np.load('data/' + FLAGS.validate_on + '_image_id_to_filename.npy')[()]
    filename_to_image_id = np.load('data/' + FLAGS.validate_on + '_filename_to_image_id.npy')[()]

for f in filenames_captions_dict.keys():
    filenames_captions_dict[f] = preprocess_for_test(filenames_captions_dict[f])

for id_, f in enumerate(image_id_to_filename):
    ims_elem = str(license_) + ',' + str(url_) + ',' + str(f) + ',' + str(id_) + ',' + str(
        width_) + ',' + str(date_captured) + ',' + str(height_)
    ims.append(ims_elem)
    for k in range(TOTAL_CAPTIONS_FOR_AN_IMAGE):
        anns_elem = str(id_) + ',' + str(randint(4000, 9000)) + ',' + str(filenames_captions_dict[f][k])
        anns.append(anns_elem)

d = {"images": [{'license': elem.split(',')[0], "url": elem.split(',')[1], "file_name": elem.split(',')[2],
                 "id": str(elem.split(',')[3]), "width": elem.split(',')[4], "date_captured": elem.split(',')[5],
                 "height": elem.split(',')[6]} for elem in ims],
     "annotations": [{'image_id': str(elem.split(',')[0]), "id": elem.split(',')[1], "caption": elem.split(',')[2]} for
                     elem in anns],
     "type": "captions",
     "info": {},
     "licenses": []}

# actually it is the test baseline
json.dump(d, open(output_path + './flikr.json', 'w'))