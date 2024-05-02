# run this file to generate interesting custom inputs
import random

if __name__ == "__main__":
    # a clique of 4000 nodes with some random dropout
    NAME = "clique4000_01"
    num_nodes = 4000
    num_edges = num_nodes * (num_nodes - 1)

    with open(f"{NAME}.txt", "w") as file:
        # Write header information
        file.write(f"# Directed graph (each unordered pair of nodes is saved once): {NAME}.txt\n")
        file.write(f"# Nodes: {num_nodes} Edges: {num_edges}\n")
        file.write("# FromNodeId\tToNodeId\n")
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() > 0.1:
                    file.write(f"{i}\t{j}\n")
