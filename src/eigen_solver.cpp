#include "eigen_solver.hpp"

#include<iostream>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>

Eigen::MatrixXd compute_first_k_eigenvectors(const Matrix& L, int n, int k) {
    // Spectra works with column-major internally
    Eigen::MatrixXd L_col = L;  
    
    // Operation object: performs matrix-vector products
    Spectra::DenseSymMatProd<double> op(L_col);

    // ncv = size of Krylov subspace (must be > k)
    int ncv = std::min(n, k * 2);

    // Compute k smallest eigenvalues (and eigenvectors)
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigen_solver(op, k, ncv);

    eigen_solver.init();
    int converged_eigenvalues = eigen_solver.compute(Spectra::SortRule::SmallestAlge);

    if (eigen_solver.info() != Spectra::CompInfo::Successful || converged_eigenvalues < k) {
        std::cerr << "Spectra eigensolver failed, converged: " << converged_eigenvalues << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return eigen_solver.eigenvectors();
}