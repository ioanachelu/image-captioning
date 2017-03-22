import json
import codecs
from random import randint
from utils import get_caption_data

TOTAL_CAPTIONS_FOR_AN_IMAGE = 5  # total captions for an image in the dataset

output_path = './data/'

feats, captions, filenames_to_captions = get_caption_data(mode="train")


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

for j, f in enumerate(filenames_captions_dict.keys()):

    ims_elem = str(license_) + ',' + str(url_) + ',' + str(f) + ',' + str(id_) + ',' + str(
        width_) + ',' + str(date_captured) + ',' + str(height_)
    ims.append(ims_elem)
    for k in range(TOTAL_CAPTIONS_FOR_AN_IMAGE):
        anns_elem = str(j) + ',' + str(randint(4000, 9000)) + ',' + str(filenames_captions_dict[f][k])
        anns.append(anns_elem)

    id_ += 1

d = {"images": [{'license': elem.split(',')[0], "url": elem.split(',')[1], "file_name": elem.split(',')[2],
                 "id": str(elem.split(',')[3]), "width": elem.split(',')[4], "date_captured": elem.split(',')[5],
                 "height": elem.split(',')[6]} for elem in ims],
     "annotations": [{'image_id': str(elem.split(',')[0]), "id": elem.split(',')[1], "caption": elem.split(',')[2]} for
                     elem in anns],
     "type": "captions",
     "info": {},
     "licenses": []}

# actually it is the test baseline
json.dump(d, open(output_path + './flickr30k_base_baseline.json', 'w'))