#include "k_means.hpp"

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

std::vector<int> evaluate_labels(const Matrix& X, const Matrix& centroids, int l, int r) {
    std::vector<int> labels(r - l);

    #pragma omp parallel for schedule(static) // pre-divide iterations into chunks
    for (int i = l; i < r; ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int label = -1;

        for (int j = 0; j < centroids.rows(); ++j) {
            double distance = (X.row(i) - centroids.row(j)).squaredNorm();

            if (distance < min_distance) {
                min_distance = distance;
                label = j;
            }
        }

        labels[i - l] = label;
    }

    return labels;
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
        local_labels = evaluate_labels(X, global_centroids, l, r);

        MPI_Gather(local_labels.data(), count, MPI_INT, global_labels.data(), count, MPI_INT, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            // calculate new centroids
            Matrix new_centroids = Matrix::Zero(k, m);
            std::vector<int> sizes(k, 0);

            #pragma omp parallel
            {
                Matrix local_centroids = Matrix::Zero(k, m);
                std::vector<int> local_sizes(k, 0);

                #pragma omp for nowait
                for (int i = 0; i < n; ++i) {
                    int c = global_labels[i];
                    local_centroids.row(c) += X.row(i);
                    local_sizes[c] += 1;
                }

                #pragma omp critical
                {
                    new_centroids += local_centroids;

                    for (int i = 0; i < k; ++i) {
                        sizes[i] += local_sizes[i];
                    }
                }
            }

            for (int i = 0; i < k; ++i) {
                if (sizes[i] > 0) {
                    new_centroids.row(i) /= static_cast<double>(sizes[i]);
                }
            }

            // split empty clusters
            for (int i = 0; i < k; ++i) {
                if (sizes[i] == 0) {
                    int largest = std::distance(sizes.begin(), std::max_element(sizes.begin(), sizes.end()));

                    double max_distance = -1.0;
                    int farthest = -1;

                    for (int j = 0; j < n; ++j) {
                        if (global_labels[j] == largest) {
                            double distance = (X.row(j) - global_centroids.row(largest)).squaredNorm();

                            if (distance > max_distance) {
                                max_distance = distance;
                                farthest = j;
                            }
                        }
                    }

                    new_centroids.row(i) = X.row(farthest);
                    sizes[i] = 1;
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