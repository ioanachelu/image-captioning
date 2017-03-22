#!/usr/bin/env bash
# Location to save the Inception v3 checkpoint.
EMBEDDING_DIR="./embedding_models"
mkdir -p ${EMBEDDING_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${EMBEDDING_DIR}
rm "inception_v3_2016_08_28.tar.gz"