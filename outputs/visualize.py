import matplotlib.pyplot as plt

def plot_data(filename):
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file]

    x_values = list(range(1, len(data) + 1))

    plt.figure(figsize=(8, 5))
    plt.bar(x_values, data, color='royalblue', width=1)
    plt.title('Hash Utilization for each kernel launch')
    plt.xlabel('# Kernel launch')
    plt.ylabel('# Hash Utilization (percent)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.savefig('../outputs/hash_utils.png')

plot_data('../outputs/output_visualize.txt')
