#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <unistd.h>

#include "mode.h"

void jones_plassmann(int node_cnt, int edge_cnt, int* colors, int *nbrs_start, int *nbrs, Mode mode, bool verbose,
    std::string strategy);
void printCudaInfo();


void compress(const std::vector<std::vector<int>>&& graph, int* nbrs_start, int* nbrs) {
    int current_start = 0;
    int current_idx = 0;
    for (size_t node = 0; node < graph.size(); ++node) {
        nbrs_start[node] = current_start;
        current_start += graph[node].size();
        for (int nbr : graph[node]) {
            nbrs[current_idx++] = nbr;
        }
    }
    nbrs_start[graph.size()] = current_start;
}

void write_output(const std::string& output_filename, int* colors, size_t num_colors) {
    std::ofstream out_file(output_filename, std::fstream::out);
    if (!out_file) {
        std::cerr << "\033[1;31mUnable to open file for output: " << output_filename << "\033[0m\n";
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < num_colors; ++i) {
        out_file << colors[i] << " ";
    }
}

int main(int argc, char *argv[]) {
    std::string input_filename;
    std::string output_filename;
    std::string strategy = "minimum";
    std::string mode = "basic";
    bool verbose = false;
    int opt;
    while ((opt = getopt(argc, argv, "f:o:vm:s:")) != -1) {
        switch (opt) {
        case 'f':
            input_filename = optarg;
            break;
        case 'o':
            output_filename = optarg;
            break;
        case 'v':
            verbose = true;
            break;
        case 'm':
            mode = optarg;
            break;
        case 's':
            strategy = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f input_filename -o output (-v) -m mode[basic/minmax/multihash/basicopt] -s strategy[minimum/maximum/double/linear4]\n";
            exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (empty(input_filename) || (mode != "basic" && mode != "minmax" && mode != "basicopt" && mode != "multihash") ||
        (strategy != "minimum" && strategy != "maximum" && strategy != "double" && strategy != "linear4")) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -o output (-v) -m mode[basic/minmax/multihash/basicopt] -s strategy[minimum/maximum/double/linear4]\n";
        exit(EXIT_FAILURE);
    }
    std::cout << "Input File: " << input_filename << std::endl;
    std::cout << "Output File: " << output_filename << std::endl;
    std::ifstream fin(input_filename);
    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
    }

    int node_cnt = 0, edge_cnt = 0;
    std::string graph_name;
    std::string line;
    for (int i = 0; i < 4; ++i) {
        std::getline(fin, line);
        // extract graph name
        if (i == 0) {
            std::istringstream iss(line);
            std::string tok;
            while (iss >> tok) {
                if (tok.find(".txt") != std::string::npos) {
                    graph_name = tok.substr(0, tok.size() - 4);
                    break;
                }
            }
        }
        // extract number of nodes
        if (line.find("Nodes:") != std::string::npos) {
            std::size_t pos = line.find("Nodes:") + 7;
            node_cnt = std::stoi(line.substr(pos, line.find("Edges:") - pos));
        }
        // extract number of edges
        if (line.find("Edges:") != std::string::npos) {
            std::size_t pos = line.find("Edges:") + 7;
            edge_cnt = std::stoi(line.substr(pos));
        }
    }
    std::cout << "Graph name: \u001b[35m\u001b[1m" << graph_name << "\033[0m" << std::endl;
    std::cout << "Number of nodes: " << node_cnt << std::endl;
    std::cout << "Number of edges: " << edge_cnt << std::endl;
    std::cout << "Coloring mode: \u001b[35m\u001b[1m" << mode << "\033[0m" << std::endl;
    std::cout << "Strategy: \u001b[35m\u001b[1m" << strategy << "\033[0m" << std::endl;

    // extract edges
    std::vector<std::vector<int>> graph(node_cnt);
    int from_node = -1, to_node = -1;
    for (int i = 0; i < edge_cnt; ++i) {
        fin >> from_node >> to_node;
        graph[from_node].push_back(to_node);
        graph[to_node].push_back(from_node);
    }

    // output colors
    int* colors = (int*)malloc(sizeof(int) * node_cnt);
    memset(colors, 0, sizeof(int) * node_cnt);
    // compressed graph
    int* nbrs = (int*)malloc(sizeof(int) * edge_cnt * 2);
    int* nbrs_start = (int*)malloc(sizeof(int) * node_cnt + 1);
    compress(std::move(graph), nbrs_start, nbrs);

    Mode m = Basic;
    if (verbose) printCudaInfo();
    if (mode == "minmax") m = MinMax;
    if (mode == "multihash") m = MultiHash;
    if (mode == "basicopt") m = BasicOpt;
    
    const auto compute_start = std::chrono::steady_clock::now();

    jones_plassmann(node_cnt, edge_cnt, colors, nbrs_start, nbrs, m, verbose, strategy);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " <<  "\033[33m" << compute_time << "\033[0m\n";

    write_output(output_filename, colors, node_cnt);
    return 0;
}