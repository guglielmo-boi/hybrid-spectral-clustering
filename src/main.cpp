#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>

#include "common.hpp"
#include "csv.hpp"
#include "spectral_clustering.hpp"

int main(int argc, char** argv) 
{
    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    if (argc < 3) {
        std::cerr << "Usage: ./program input_file.csv output_file.csv" << std::endl;
        return 1;
    }

    Matrix X;
    std::vector<int> labels;

    if (!load_csv(input_path, X, labels)) {
        std::cerr << "Error: cannot open input file at path " << argv[1] << std::endl;
        return 1;
    }
    
    int max_label = *std::max_element(labels.begin(), labels.end());
    
    std::vector<int> output_labels = spectral_clustering(X, max_label + 1);

    if (!save_csv(output_path, X, output_labels)) {
        std::cerr << "Error: cannot open output file at path " << output_path << std::endl;
        return 1;
    }

    return 0;
}
