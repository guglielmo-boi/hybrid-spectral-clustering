#include "spectral_clustering.hpp"

#include "k_means.hpp"

std::vector<double> evaluate_gaussian_similarity_values(const Matrix& X, int l, int r, double sigma) {
    std::vector<double> similarity_values;
    
    const double denominator = 2 * sigma * sigma;

    for (int i = l; i < r; ++i) {
        for (int j = 0; j < X.rows(); ++j) {
            if (i != j) {
                double squared_euclidean_distance = (X.row(i) - X.row(j)).squaredNorm();
                double similarity = exp(-squared_euclidean_distance / denominator);
                similarity_values.push_back(similarity);
            } else {
                similarity_values.push_back(0.0);
            }
        }
    }

    return similarity_values;
}

std::vector<double> evaluate_diagonal_values(const Eigen::VectorXd& degrees, int l, int r) {
    std::vector<double> diagonal_values;

    for (int i = l; i < r; ++i) {
        if (degrees(i) > 1e-12) {
            diagonal_values.push_back(1.0 / sqrt(degrees(i)));
        } else {
            diagonal_values.push_back(0.0);
        }
    }

    return diagonal_values;
}

std::vector<int> spectral_clustering(Matrix& X, int k, double sigma) {
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = X.rows();
    int count = n / world_size; // number of rows for each process
    int l = count * world_rank; // index of the local begin row
    int r = count * (world_rank + 1); // index of the local end row

    Matrix local_eigenvectors = Matrix::Zero(count, k);
    Matrix global_eigenvectors = Matrix::Zero(n, k);

    std::vector<double> local_similarity_values = evaluate_gaussian_similarity_values(X, l, r, sigma);

    if (world_rank != 0) {
        // similarity matrix
        MPI_Gather(local_similarity_values.data(), count * n, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

        // diagonal matrix
        Eigen::VectorXd degrees = Eigen::VectorXd::Zero(n);    
        MPI_Bcast(degrees.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> local_diagonal_values = evaluate_diagonal_values(degrees, l, r);
        MPI_Gather(local_diagonal_values.data(), count, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        // similarity matrix
        std::vector<double> global_similarity_values(n * n);
        MPI_Gather(local_similarity_values.data(), count * n, MPI_DOUBLE, global_similarity_values.data(), count * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Matrix similarity_matrix = Eigen::Map<Matrix>(global_similarity_values.data(), n, n);
    
        // diagonal matrix
        Eigen::VectorXd degrees = similarity_matrix.rowwise().sum();
        MPI_Bcast(degrees.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> local_diagonal_values = evaluate_diagonal_values(degrees, l, r);

        std::vector<double> global_diagonal_values(n);
        MPI_Gather(local_diagonal_values.data(), count, MPI_DOUBLE, global_diagonal_values.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Eigen::VectorXd diagonal_vector = Eigen::Map<Eigen::VectorXd>(global_diagonal_values.data(), n);

        // normalized graph Laplacian
        Matrix L = diagonal_vector.asDiagonal() * similarity_matrix * diagonal_vector.asDiagonal();
        L = Matrix::Identity(n, n) - L;

        // eigen decomposition
        Eigen::SelfAdjointEigenSolver<Matrix> solver(L);
        global_eigenvectors = solver.eigenvectors().leftCols(k);
    }

    // eigenvectors normalization
    MPI_Scatter(global_eigenvectors.data(), count * k, MPI_DOUBLE, local_eigenvectors.data(), count * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < count; ++i) {
        local_eigenvectors.row(i).normalize();
    }

    MPI_Allgather(local_eigenvectors.data(), count * k, MPI_DOUBLE, global_eigenvectors.data(), count * k, MPI_DOUBLE, MPI_COMM_WORLD);

    return k_means(global_eigenvectors, k);
}