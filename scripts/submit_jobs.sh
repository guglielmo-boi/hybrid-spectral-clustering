#!/bin/bash

DATASETS=("gaussian" "mixed")
SIZES=(8192 32768 131072)
RANKS_LIST=(1 2 4 8 16 32)

TEMPLATE="job_template.pbs"

for DATA in "${DATASETS[@]}"; do

  # Short prefix for job name
  if [ "$DATA" = "gaussian" ]; then
    PREFIX="g"
  else
    PREFIX="m"
  fi

  for SIZE in "${SIZES[@]}"; do
    for RANKS in "${RANKS_LIST[@]}"; do

      JOB_FILE="job_${DATA}_${SIZE}_r${RANKS}.pbs"
      JOB_NAME="${PREFIX}_${SIZE}_r${RANKS}"

      sed \
        -e "s/__RANKS__/${RANKS}/" \
        -e "s/\${DATA}/${DATA}/g" \
        -e "s/\${SIZE}/${SIZE}/g" \
        -e "s/\${RANKS}/${RANKS}/g" \
        "$TEMPLATE" > "$JOB_FILE"

      echo "Submitting $JOB_NAME"
      qsub -N "$JOB_NAME" "$JOB_FILE"

    done
  done
done
