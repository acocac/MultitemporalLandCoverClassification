#!/bin/bash

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Convenience script for training model locally.
#
# Arguments:
#   MODEL_INPUTS_DIR: The directory containing the TFRecords from preprocessing.
#                     This should just be a timestamp.
. ./bin/_common.sh

MODEL_DIR=./model
INPUT_PATH=./data_IJGI18_demo/data_IJGI18/datasets/demo/${PZISE}

EPOCHS=1
TRAIN_YEAR=2016

CELL=(4)
LAYERS=(1)
LR=(0.001)
BS=(8)

gcloud ai-platform local train \
  --module-name trainer.task \
  --package-path trainer \
  -- \
  --modeldir "${MODEL_DIR}" \
  --datadir "${INPUT_PATH}" \
  --train_on "${TRAIN_YEAR}" \
  --epochs "${EPOCHS}" \
  --convrnn_filters ${CELL} \
  --convrnn_layers ${LAYERS} \
  --learning_rate "${LR}" \
  --batchsize ${BS}