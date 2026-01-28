#!/bin/bash
#PBS -l select=__RANKS__:ncpus=4:mem=2gb
#PBS -l place=scatter
#PBS -l walltime=06:00:00
#PBS -q shortCPUQ

cd $PBS_O_WORKDIR

# OpenMP settings
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

INPUT=$HOME/hpc-project/data/input/${DATA}_${SIZE}.csv
OUTPUT=$HOME/hpc-project/data/output/${DATA}_${SIZE}_r${RANKS}.csv

mpiexec --mca mpi_cuda_support 0 -n ${RANKS} $HOME/hpc-project/bin/spectral_clustering "$INPUT" "$OUTPUT"
