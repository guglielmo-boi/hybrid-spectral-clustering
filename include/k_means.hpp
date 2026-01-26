#ifndef K_MEANS_HPP
#define K_MEANS_HPP

#include "common.hpp"

#include <vector>

std::vector<int> k_means(const Matrix& X, int k, int max_iters = 1000);

#endif