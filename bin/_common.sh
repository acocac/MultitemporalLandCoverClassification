#!/bin/bash

SCALE_TIER="CUSTOM"
MODEL_NAME="mtlcc"
HPTUNING_CONFIG="trainer/hptuning_config.yaml"
BUCKET_NAME="gcptutorials"
PROJECT="sample"
PZISE=240
REGION="us-west1"
#export TF_FORCE_GPU_ALLOW_GROWTH=true

function get_date_time {
  echo "$(date +%Y%m%d%H%M%S)"
}