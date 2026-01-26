#!/bin/bash

LOCAL_DIR="/home/guglielmo/Uni/hpc-project/data/output/"
REMOTE_USER="guglielmo.boi"
REMOTE_HOST="hpc3-login.unitn.it"
REMOTE_DIR="/home/guglielmo.boi/hpc-project/data/output/"

mkdir -p "$LOCAL_DIR"

scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}*" "$LOCAL_DIR"
