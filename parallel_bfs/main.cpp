#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

#include <unistd.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    std::string input_filename;
    int opt;
    while ((opt = getopt(argc, argv, "f:")) != -1) {
        switch (opt) {
        case 'f':
            input_filename = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
            exit(EXIT_FAILURE);
        }
    }
    // Check if required options are provided
    if (empty(input_filename)) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename\n";
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
                    graph_name = tok;
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
    std::cout << "Graph name: " << graph_name << std::endl;
    std::cout << "Number of nodes: " << node_cnt << std::endl;
    std::cout << "Number of edges: " << edge_cnt << std::endl;

    // extract edges
    std::vector<std::vector<int>> graph;
    graph.resize(node_cnt);
    int from_node = -1, to_node = -1;
    for (int i = 0; i < edge_cnt; ++i) {
        fin >> from_node >> to_node;
        graph[from_node].push_back(to_node);
    }
    return 0;
}