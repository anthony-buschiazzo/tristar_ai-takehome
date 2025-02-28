#!/bin/sh

#shellcheck disable=SC2034
#shellcheck disable=SC1143

model_dir="/home/tristarAI/takehome_ws/trained_models/default_config"
model="bestRecall.pth"

python3 evaluate.py     --model_dir $model_dir \
                        --model $model