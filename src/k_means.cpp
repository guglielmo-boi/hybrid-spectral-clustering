#include "k_means.hpp"

#include "similarity_matrix.hpp"

#include <random>
#include <limits>
#include <cmath>

Matrix init_centroids(const Matrix& X, int k) {
    Matrix centroids(k, X.cols());

    std::default_random_engine rng(100);
    std::uniform_int_distribution<> dist(0, X.rows() - 1);
    
    for (int i = 0; i < k; ++i) {
        centroids.row(i) = X.row(dist(rng));
    }

    return centroids;
}

std::vector<int> k_means(const Matrix& X, int k, int max_iters) {
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = X.rows();
    int m = X.cols();
    int count = n / world_size; // number of rows for each process
    int l = count * world_rank; // index of the local begin row
    int r = count * (world_rank + 1); // index of the local end row

    Matrix global_centroids(k, X.cols());

    if (world_rank == 0) {
        global_centroids = init_centroids(X, k);
    }

    std::vector<int> local_labels(count, -1); 
    std::vector<int> global_labels(n, -1);

    int iterating = 1;

    for (int iter = 0; iterating && iter < max_iters; ++iter) {
        MPI_Bcast(global_centroids.data(), k * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        local_labels = evaluate_k_means_labels(X, global_centroids, l, r);

        MPI_Gather(local_labels.data(), count, MPI_INT, global_labels.data(), count, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            Matrix new_centroids = Matrix::Zero(k, m);
            std::vector<int> sizes(k, 0);

            for (int i = 0; i < n; ++i) {
                int c = global_labels[i];
                new_centroids.row(c) += X.row(i);
                sizes[c] += 1;
            }

            for (int j = 0; j < k; ++j) {
                if (sizes[j] > 0) {
                    new_centroids.row(j) /= static_cast<double>(sizes[j]);
                }
            }

            for (int j = 0; j < k; ++j) {
                if (sizes[j] == 0) {
                    int largest = std::distance(sizes.begin(), std::max_element(sizes.begin(), sizes.end()));

                    double max_distance = -1.0;
                    int farthest = -1;

                    for (int i = 0; i < n; ++i) {
                        if (global_labels[i] == largest) {
                            double d = (X.row(i) - global_centroids.row(largest)).squaredNorm();

                            if (d > max_distance) {
                                max_distance = d;
                                farthest = i;
                            }
                        }
                    }

                    new_centroids.row(j) = X.row(farthest);
                    sizes[j] = 1;
                    sizes[largest] -= 1;
                }
            }


            // convergence check
            if ((new_centroids - global_centroids).norm() < 1e-3) {
                iterating = 0;
            }

            global_centroids = new_centroids;
        }

        MPI_Bcast(&iterating, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    return global_labels;
}