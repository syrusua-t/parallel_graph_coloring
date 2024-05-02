# run this file to generate interesting custom inputs

if __name__ == "__main__":
    # Define the parameters
    NAME = "clique4000"
    num_nodes = 4000
    num_edges = num_nodes * (num_nodes - 1)

    with open(f"{NAME}.txt", "w") as file:
        # Write header information
        file.write(f"# Directed graph (each unordered pair of nodes is saved once): {NAME}.txt\n")
        file.write(f"# Nodes: {num_nodes} Edges: {num_edges}\n")
        file.write("# FromNodeId\tToNodeId\n")
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    file.write(f"{i}\t{j}\n")
