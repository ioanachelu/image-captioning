import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

filenames = os.listdir("captions")
ids = [int(os.path.splitext(f)[0][3:]) for f in filenames]
id_to_image_path = {}
len(ids)

captions_json = json.load(open('data/predictions.json'))


def find_caption_for_id(id):
    for entry in captions_json:
        if entry["image_id"] == id:
            return entry["caption"]


for i, id in enumerate(ids):
    id_to_image_path[id] = os.path.join("captions", filenames[i])


for id in ids:
#     show_n_images -= 1
#     if show_n_images == 0:
#         break
    pil_im = Image.open(id_to_image_path[id]).convert('RGB')
#     imshow(pil_im)
    plt.figure()
    plt.imshow(pil_im)
    plt.suptitle(find_caption_for_id(id))
    plt.savefig('./img_captions_test/img_with_caption_' + str(id) + '.png', bbox_inches='tight')
    plt.clf()