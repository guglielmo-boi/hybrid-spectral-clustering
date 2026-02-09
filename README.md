## High Performance Computing for Data Science 2025-2026
# Hybrid Parallelization of 3D Spectral Clustering
### Guglielmo Boi

Spectral clustering is effective for identifying non-spherical clusters but suffers from high computational and memory costs that limit scalability. This work presents a hybrid parallel implementation of spectral clustering for three-dimensional data, combining MPI and OpenMP. The implementation is evaluated on synthetic datasets with up to 131,072 points. Results show that the method maintains good clustering accuracy while achieving moderate speedup, with scalability primarily limited by eigenvalue decomposition time complexity and memory bandwidth saturation.