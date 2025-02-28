#!/usr/bin/env bash

ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
ROS_HOSTNAME=${ROS_HOSTNAME:-localhost}

HOME_DIR=/home/tristarAI

docker run --gpus all -it --rm \
                --net=host \
                --ipc=host \
                -e ROS_HOSTNAME=$ROS_HOSTNAME \
                -e ROS_MASTER_URI=$ROS_MASTER_URI \
                -e DISPLAY=unix$DISPLAY \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -v /etc/localtime:/etc/localtime:ro \
                -v "$PWD"/data:/data \
                -v "$PWD"/takehome_ws:/${HOME_DIR}/takehome_ws \
                tristar-ai-takehome:latest "$@"
