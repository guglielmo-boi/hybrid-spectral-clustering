#!/bin/bash

LOCAL_OUTPUT_DIR="/home/guglielmo/Uni/hpc-project/data/output/"
LOCAL_LOGS_DIR="/home/guglielmo/Uni/hpc-project/logs/"

REMOTE_USER="guglielmo.boi"
REMOTE_HOST="hpc3-login.unitn.it"
REMOTE_OUTPUT_DIR="/home/guglielmo.boi/hpc-project/data/output/"
REMOTE_LOGS_DIR="/home/guglielmo.boi/hpc-project/logs/"

mkdir -p "$LOCAL_OUTPUT_DIR"
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_OUTPUT_DIR}*" "$LOCAL_OUTPUT_DIR"

mkdir -p "$LOCAL_LOGS_DIR"
scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_LOGS_DIR}*" "$LOCAL_LOGS_DIR"