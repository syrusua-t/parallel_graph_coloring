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

void write_output(const std::string& output_filename, const std::vector<int>& colors) {
    std::ofstream out_file(output_filename, std::fstream::out);
    if (!out_file) {
        std::cerr << "\033[1;31mUnable to open file for output: " << output_filename << "\033[0m\n";
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < colors.size(); ++i) {
        out_file << colors[i] << " ";
    }
}

int main(int argc, char *argv[]) {
    std::string input_filename;
    std::string mode = "p";
    std::string output_filename;
    int opt;
    while ((opt = getopt(argc, argv, "f:m:o:")) != -1) {
        switch (opt) {
        case 'f':
            input_filename = optarg;
            break;
        case 'm':
            mode = optarg;
            break;
        case 'o':
            output_filename = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f input_filename -m s/p\n";
            exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (empty(input_filename) || (mode != "s" && mode != "p")) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -m s/p\n";
        exit(EXIT_FAILURE);
    }
    std::cout << "Input File: " << input_filename << std::endl;
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

    // extract edges
    std::vector<std::vector<int>> graph(node_cnt);
    int from_node = -1, to_node = -1;
    for (int i = 0; i < edge_cnt; ++i) {
        fin >> from_node >> to_node;
        graph[from_node].push_back(to_node);
        graph[to_node].push_back(from_node);
    }

    // output colors
    std::vector<int> colors(node_cnt, 0);

    const auto compute_start = std::chrono::steady_clock::now();

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " <<  "\033[33m" << compute_time << "\033[0m\n";

    write_output(output_filename, colors);
    return 0;
}