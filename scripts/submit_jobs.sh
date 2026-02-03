#!/bin/bash
set -e

# --- Resolve paths ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"
TEMPLATE="$SCRIPT_DIR/job_template.pbs"

echo "=== Project root: $PROJECT_DIR ==="
echo "=== Using template: $TEMPLATE ==="

# --- Load modules needed for compilation ---
module purge
module load GCC/13.2.0 CMake/3.27.6-GCCcore-13.2.0 OpenMPI/4.1.6-GCC-13.2.0

# --- Rebuild project ---
echo "=== Recompiling project ==="
cd "$PROJECT_DIR"
rm -rf build bin
cmake -B build
cmake --build build
echo "=== Build completed ==="

# --- Experiment parameters ---
DATASETS=("gaussian")
SIZES=(16384 32768 65536)
RANKS_LIST=(1 2 4 8 16 32 64)

# --- Submit jobs ---
for DATA in "${DATASETS[@]}"; do

  if [ "$DATA" = "gaussian" ]; then
    PREFIX="g"
  else
    PREFIX="m"
  fi

  for SIZE in "${SIZES[@]}"; do
    for RANKS in "${RANKS_LIST[@]}"; do

      NCPUS=$((RANKS))
      JOB_NAME="${PREFIX}_${SIZE}_r${RANKS}"

      echo "Submitting $JOB_NAME (ncpus=$NCPUS)"

      sed \
        -e "s/__NCPUS__/${NCPUS}/" \
        -e "s/\${DATA}/${DATA}/g" \
        -e "s/\${SIZE}/${SIZE}/g" \
        -e "s/\${RANKS}/${RANKS}/g" \
        "$TEMPLATE" | qsub -N "$JOB_NAME"

    done
  done
done

echo "=== All jobs submitted successfully ==="
