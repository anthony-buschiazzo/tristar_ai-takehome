#!/usr/bin/env bash

set -eo pipefail

DOCKERFILE_DIR=$(dirname -- "$( readlink -f -- "$0"; )")
CONTEXT_DIR=$(dirname $DOCKERFILE_DIR)

docker build -f ${DOCKERFILE_DIR}/Dockerfile -t tristar-ai-takehome:latest ${CONTEXT_DIR}

