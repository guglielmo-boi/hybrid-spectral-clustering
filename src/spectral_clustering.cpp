#include "spectral_clustering.hpp"

#include "k_means.hpp"

Matrix gaussian_similarity_matrix(const Matrix& matrix, double sigma) {
    int n = matrix.rows();
    Matrix similarity_matrix = Matrix::Zero(n, n);

    double denominator = 2 * sigma * sigma;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double squared_euclidean_distance = (matrix.row(i) - matrix.row(j)).squaredNorm();
            double similarity = exp(-squared_euclidean_distance / denominator);
            
            similarity_matrix(i, j) = similarity;
            similarity_matrix(j, i) = similarity;
        }
    }

    return similarity_matrix;
}

std::vector<int> spectral_clustering(const Matrix& matrix, int k, double sigma) {
    int n = matrix.rows();
    Matrix similarity_matrix = gaussian_similarity_matrix(matrix, sigma);

    // diagonal matrix
    Eigen::VectorXd degrees = similarity_matrix.rowwise().sum();
    Matrix diagonal_matrix = Matrix::Zero(n, n);

    for (int i = 0; i < n; ++i) {
        if (degrees(i) > 1e-12) {
            diagonal_matrix(i, i) = 1.0 / sqrt(degrees(i));
        }
    }

    // normalized graph Laplacian
    Matrix L = Matrix::Identity(n, n) - diagonal_matrix * similarity_matrix * diagonal_matrix;

    // eigen decomposition
    Eigen::SelfAdjointEigenSolver<Matrix> solver(L);
    Matrix eigenvectors = solver.eigenvectors().leftCols(k);

    // normalize rows
    for (int i = 0; i < n; ++i) {
        eigenvectors.row(i).normalize();
    }

    return k_means(eigenvectors, k);
}