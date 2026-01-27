#!/bin/bash
#PBS -l select=8:ncpus=16:mem=2gb -l place=pack
#PBS -l walltime=0:30:00
#PBS -q shortCPUQ

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

module load GCC/13.2.0 CMake/3.27.6-GCCcore-13.2.0 OpenMPI/4.1.6-GCC-13.2.0
mpiexec --mca mpi_cuda_support 0 -n 8 ./hpc-project/bin/spectral_clustering ./hpc-project/data/input/gaussian_8192.csv ./hpc-project/data/output/gaussian_8192.csv
