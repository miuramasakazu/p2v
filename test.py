import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


f = open('res.txt')
all_embeddings = []
all_programs = []
program2id = dict()

for i, line in enumerate(f):
    line = line.strip().split(' ')
    program = line[0]
    embedding = [float(x) for x in line[1:]]
    all_embeddings.append(embedding)
    all_programs.append(program)
    program2id[program] = i
all_embeddings = np.array(all_embeddings)

while 1:
    p = input('Program(enter q to Quit):')
    if p == 'q':
        break
    try:
        pid = program2id[p]
    except:
        print('Cannot find this program')
        continue

    embedding = all_embeddings[pid: pid+1]
    d = cosine_similarity(embedding, all_embeddings)[0]
    d = zip(all_programs, d)
    d = sorted(d, key=lambda x: x[1], reverse=True)
    for p in d[1:11]:
        if len(p[0]) < 2:
            continue
        print(p)
