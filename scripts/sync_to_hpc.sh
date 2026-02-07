#!/bin/bash

LOCAL_DIR="/home/guglielmo/Uni/hpc-project/"
REMOTE_USER="guglielmo.boi"
REMOTE_HOST="hpc3-login.unitn.it"
REMOTE_DIR="/home/guglielmo.boi/hpc-project/"

rsync -avz --delete \
  --exclude=".git/" \
  --exclude=".venv/" \
  --exclude=".vscode/" \
  --exclude="bin/" \
  --exclude="build/" \
  --exclude="data/output/" \
  --exclude="logs/"
  "$LOCAL_DIR" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
