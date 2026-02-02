#include "common.hpp"
#include "csv.hpp"
#include "spectral_clustering.hpp"

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>

int main(int argc, char** argv) 
{
    int provided_threading;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_threading);
    Eigen::setNbThreads(omp_get_max_threads());

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // PARALLEL ENVIRONMENT INFO
    int omp_max_threads = omp_get_max_threads();
    int omp_num_threads = 0;

    #pragma omp parallel
    {
        #pragma omp master
        omp_num_threads = omp_get_num_threads();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < world_size; ++r) {
        if (r == world_rank) {
            std::cout << "[Rank " << world_rank << "] "
                    << "MPI ranks total: " << world_size
                    << " | OMP max threads: " << omp_max_threads
                    << " | OMP active threads: " << omp_num_threads
                    << " | Eigen threads: " << Eigen::nbThreads()
                    << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    Matrix X;
    int rows = 0, cols = 0;
    std::vector<int> labels;
    int max_label = 0;

    if (world_rank == 0) {
        if (argc < 3) {
            std::cerr << "Usage: ./program input_file.csv output_file.csv" << std::endl;
            return 1;
        }

        if (!load_csv(input_path, X, labels)) {
            std::cerr << "Error: cannot open input file at path " << argv[1] << std::endl;
            return 1;
        }
        
        rows = X.rows();
        cols = X.cols();
        max_label = *std::max_element(labels.begin(), labels.end());
    } 

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_label, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (world_rank != 0) {
        X = Matrix::Zero(rows, cols);
    }

    MPI_Bcast(X.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    std::vector<int> output_labels = spectral_clustering(X, max_label + 1);

    if (world_rank == 0) {        
        if (!save_csv(output_path, X, output_labels)) {
            std::cerr << "Error: cannot open output file at path " << output_path << std::endl;
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    return 0;
}
