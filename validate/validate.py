import sys

help_message = '''usage: validate.py [-i input_file] [-c color_output]
'''

def parse_args():
    args = sys.argv
    if '-h' in args or '--help' in args:
        print (help_message)
        sys.exit(1)
    if '-i' not in args or '-c' not in args:
        print (help_message)
        sys.exit(1)
    parsed = {}
    parsed['input'] = args[args.index('-i') + 1]
    parsed['output'] = args[args.index('-c') + 1]
    return parsed

def parse_input(input_file):
    input = open(input_file, 'r')
    ans = {'edges':[], 'edge_cnt': 0, 'node_cnt': 0}
    for line in input:
        if line.startswith('#'):
            if "Nodes:" in line:
                parts = line.split()
                ans['node_cnt'] = int(parts[2])
                ans['edge_cnt'] = int(parts[4])
        else:
            from_node, to_node = map(int, line.strip().split())
            ans['edges'].append((from_node, to_node))
    return ans

def parse_output(output_file):
    with open(output_file) as output:
        colors = [int(num) for num in output.readline().strip().split()]
        num_colors = len(set(colors))
        return num_colors, colors

def main(args):
    input = parse_input(args['input'])
    num_colors, colors = parse_output(args['output'])
    conflict = 0
    for edge in input['edges']:
        if colors[edge[0]] == colors[edge[1]]:
            conflict = conflict + 1
    if (conflict > 0):
        print("validation \033[1;31m FAILED\033[0m\n#conflicts:", conflict)
    else:
        print("validation \033[1;32mSUCCEEDED\033[0m\n#colors used:", num_colors)

if __name__ == '__main__':
    main(parse_args())