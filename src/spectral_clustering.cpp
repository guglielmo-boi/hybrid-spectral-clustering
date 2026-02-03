#include "spectral_clustering.hpp"

#include "eigen_solver.hpp"
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

std::vector<int> spectral_clustering(const Matrix& X, int k, double sigma) {
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = X.rows();
    int count = n / world_size; // number of rows for each process
    int l = count * world_rank; // index of the local begin row
    int r = count * (world_rank + 1); // index of the local end row

    Matrix local_eigenvectors(count, k);
    Matrix global_eigenvectors(n, k);

    std::vector<double> local_similarity_values = evaluate_gaussian_similarity_values(X, l, r, sigma);

    if (world_rank != 0) {
        // similarity matrix (gather one line at the time)
        for (int c = 0; c < count; ++c) {
            MPI_Gather(local_similarity_values.data() + c * n, n, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        local_similarity_values = {}; // clear and free memory

        // degrees matrix as diagonal vector
        Eigen::VectorXd degrees(n);    
        MPI_Bcast(degrees.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<double> local_diagonal_values = evaluate_diagonal_values(degrees, l, r);
        degrees.resize(0); // clear and free memory

        MPI_Gather(local_diagonal_values.data(), count, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        local_diagonal_values = {}; // clear and free memory
    } else {
        // similarity matrix (gather one line at the time)
        Matrix similarity_matrix(n, n);

        std::vector<double> global_similarity_values(world_size * n);

        for (int c = 0; c < count; ++c) {
            MPI_Gather(local_similarity_values.data() + c * n, n, MPI_DOUBLE, global_similarity_values.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (int r = 0; r < world_size; ++r) {
                int offset = r * count + c;
                std::memcpy(similarity_matrix.data() + offset * n, global_similarity_values.data() + r * n, n * sizeof(double));
            }
        }

        local_similarity_values = {}; // clear and free memory

        // degrees matrix as diagonal vector
        Eigen::VectorXd degrees = similarity_matrix.rowwise().sum();
        MPI_Bcast(degrees.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        std::vector<double> local_diagonal_values = evaluate_diagonal_values(degrees, l, r);
        degrees.resize(0); // clear and free memory
        
        Eigen::VectorXd degrees_vector(n);
        MPI_Gather(local_diagonal_values.data(), count, MPI_DOUBLE, degrees_vector.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        local_diagonal_values = {}; // clear and free memory

        // normalized graph Laplacian
        Matrix L = degrees_vector.asDiagonal() * similarity_matrix * degrees_vector.asDiagonal();
        L = Matrix::Identity(n, n) - L;

        // make sure L is symmetric
        L = 0.5 * (L + L.transpose());

        // compute first k eigenvectors
        global_eigenvectors = compute_first_k_eigenvectors(L, n, k);
    }

    // eigenvectors normalization
    MPI_Scatter(global_eigenvectors.data(), count * k, MPI_DOUBLE, local_eigenvectors.data(), count * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < count; ++i) {
        local_eigenvectors.row(i).normalize();
    }

    MPI_Allgather(local_eigenvectors.data(), count * k, MPI_DOUBLE, global_eigenvectors.data(), count * k, MPI_DOUBLE, MPI_COMM_WORLD);

    return k_means(global_eigenvectors, k);
}