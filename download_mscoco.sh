#!/bin/bash
if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
}

# Download the images.
BASE_IMAGE_URL="http://msvocds.blob.core.windows.net/coco2014"

TRAIN_IMAGE_FILE="train2014.zip"
download_and_unzip ${BASE_IMAGE_URL} ${TRAIN_IMAGE_FILE}
TRAIN_IMAGE_DIR="./mscoco/train2014"

VAL_IMAGE_FILE="val2014.zip"
download_and_unzip ${BASE_IMAGE_URL} ${VAL_IMAGE_FILE}
VAL_IMAGE_DIR="./mscoco/val2014"

# Download the captions.
BASE_CAPTIONS_URL="http://msvocds.blob.core.windows.net/annotations-1-0-3"
CAPTIONS_FILE="captions_train-val2014.zip"
download_and_unzip ${BASE_CAPTIONS_URL} ${CAPTIONS_FILE}
TRAIN_CAPTIONS_FILE="./mscoco/annotations/captions_train2014.json"
VAL_CAPTIONS_FILE="./mscoco/annotations/captions_val2014.json"
