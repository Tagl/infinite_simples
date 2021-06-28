from collections import defaultdict
res = defaultdict(list)

with open('infinite_S4.txt', 'r') as f:
    for line in f:
        line = line.strip()
        res[line.count('_')+1].append(line)


for k in res:
    filename = 'infinite_S4/length'+str(k)
    with open(filename, 'w') as f:
        for line in res[k]:
            f.write("{}\n".format(line))


