import matplotlib.pyplot as plt

def plot_data(filename):
    with open(filename, 'r') as file:
        utils = []
        progs = []
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                util, prog = map(float, parts)
                utils.append(util * 100)
                progs.append(prog)

    it = list(range(1, len(utils) + 1))

    plt.figure(figsize=(8, 5))
    plt.bar(it, utils, label='hash util score', color='royalblue', width=1)
    it.insert(0, 0)
    progs.insert(0, 0)
    plt.plot(it, progs, label='Coloring progress', linestyle='-', color='red')
    plt.title('Hash Utilization for each kernel launch')
    plt.xlabel('# Kernel launch')
    plt.ylabel('Coloring Progress/Hash Utilization Score')
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.savefig('../outputs/hash_utils.png')

plot_data('../outputs/output_visualize.txt')
