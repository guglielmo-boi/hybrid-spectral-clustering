#include "eigen_solver.hpp"

#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

Eigen::MatrixXd eigen_solver(const Matrix& L, int n, int k) {
    // Spectra works with column-major internally
    Eigen::MatrixXd L_col = L;  
    Spectra::DenseSymMatProd<double> op(L_col);

    // ncv = size of Krylov subspace (must be > k)
    int ncv = std::min(n, std::max(2 * k + 1, 20));

    // Compute k smallest eigenvalues (and eigenvectors)
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, k, ncv);

    eigs.init();
    eigs.compute(Spectra::SortRule::SmallestAlge);

    return eigs.eigenvectors();
}