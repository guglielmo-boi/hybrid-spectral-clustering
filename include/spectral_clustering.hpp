#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include "common.hpp"

#include <vector>

std::vector<int> spectral_clustering(const Matrix& X, int k, double sigma = 1.0);

#endif