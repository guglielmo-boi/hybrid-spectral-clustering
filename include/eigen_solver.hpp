#ifndef EIGEN_SOLVER_HPP
#define EIGEN_SOLVER_HPP

#include "common.hpp"

Eigen::MatrixXd compute_first_k_eigenvectors(const Matrix& L, int n, int k);

#endif