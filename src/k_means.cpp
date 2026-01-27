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

std::vector<int> k_means(const Matrix& X, int k, int max_iters) {
    int n = X.rows();
    int m = X.cols();

    Matrix centroids = init_centroids(X, k);
    std::vector<int> labels(X.rows(), 0);

    for (int iter = 0; iter < max_iters; ++iter) {
        for (int i = 0; i < X.rows(); ++i) {
            double min_distance = std::numeric_limits<double>::max();
            int label = -1;

            for (int j = 0; j < k; ++j) {
                double distance = (X.row(i) - centroids.row(j)).squaredNorm();

                if (distance < min_distance) {
                    min_distance = distance;
                    label = j;
                }
            }

            labels[i] = label;
        }

        Matrix new_centroids = Matrix::Zero(k, m);
        std::vector<int> sizes(k, 0);

        // calculate new centroids
        for (int i = 0; i < n; ++i) {
            int c = labels[i];
            new_centroids.row(c) += X.row(i);
            sizes[c] += 1;
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
                    if (labels[j] == largest) {
                        double distance = (X.row(j) - centroids.row(largest)).squaredNorm();

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
        if ((new_centroids - centroids).norm() < 1e-3) {
            break;
        }

        centroids = new_centroids;
    }

    return labels;
}