#!/bin/sh

#shellcheck disable=SC2034
#shellcheck disable=SC1143

config_file="/home/tristarAI/takehome_ws/configs/default.yaml"

python3 train.py    --config_file $config_file