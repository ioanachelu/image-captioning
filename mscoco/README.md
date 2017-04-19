# Image Captioning

This is an implementation of a "Show and Tell" like model using VGG-16 as an encoder network for the images and an
LSTM network as decoder for the captions

## System requirements
* Python 3.5
* Hdf
* a GPU - I train on an NVIDIA 1070 with batch size 16

## Python requirements
* provided you are located in the root directory, just run:
        
        $ sudo pip install -r requirements.txt
* any additional requirements that might have been omitted and prevent running the training or eval procedure can be later installed using:
        
        $ sudo pip install <requirement_name>

## Usage

### Options
* a list of all the available command line options can be found using:
        
        $ python train.py -h
* all the options can be configured using ```--<opton_name>=<option_value>``` as command line arguments or by directly modifying the flags.py file 

### Preprocessing
* the first step is downloading the dataset from mscoco website. Please download both train and validation data, as well as annotations from here:
    * http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    * http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    * http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
* the train and validation data have to be unzipped in the root. That should generate folders ```train2014``` and ```val2014```
* the annotations have to be stored in a directory named ```annotations``` that should contain ```captions_train2014.json``` and ```captions_val2014.json```
* the second step is running the preprocessing of the dataset. Please first make sure you create empty directories in the root names: ```data```, ```captions```, 
```img_captions_test```
        
        $ python preprocessing.py
* this should create that compact dataset in ```data/coco.h5``` format and the annotation files: ```data/coco.json``` and ```coco_raw.json```
* you should also download the vgg pre-trained weights from here: https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM and store them in the data folder

### Training
* for training a new model use the following command and make sure you use --train=True and --resume=False or specify so in flags.py as defaults
        
        $ python train.py 
* for resuming training you can use the ```--resume=True``` flag


### Validation
* validation is performed every ```--checkpoint_every=<nr_of_steps>``` which can be specify as command line argument or in flags.py

### Evaluation
* after training models can be evaluated using:
        
        $ python eval.py
* this takes the last model from the checkpoint directory specified in the flags.py file or given as command line argument and evaluates it on the test
dataset giving the results for the metrics at the end saving in ```data/predictions.json``` the results for each image in the test dataset. Images are saved 
at a new location: in the ```captions``` folder previously created as empty
* for an image-caption association, you can run:
        
        $ python compute_imgs_pred.py
* this will save a new image for each of the images in the ```captions``` folder in the ```imgs_captions_test``` folder previously created. These new saved images have
 as title the caption describing each of them
 
 * you can evaluate with greedy decoding by specifying ```--beam_search_size=1``` or with beam seach by using ```--beam_search_size=1```

### Tensorboard visualizations
* during training you can follow the progress using tensorboard visualizations by running from the summaries directory 
specified in the flags.py file or
given as command line argument:

        $ tensorboard --logdir=.
        
* watch the training visualizations at ```localhost:6006```
* you can also specify how many steps in between summaries by using the flag ```--summaries_every```


