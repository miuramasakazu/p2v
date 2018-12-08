import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def program_info(program, s, program_df):
    program_head = program_df[program_df["番組コード"] == program].mode().iloc[0]
    view = ["局コード", "曜日コード", "開始時刻", "終了時刻"]
    print(program_head["番組名（漢字）"], s)
    for v in  view:
        print(v, program_head[v], end=" ")
    print()
    print("------------------------------------------------------")

columns_names = ['局コード', '番組コード', '曜日コード',
                '放送日', '開始時刻', '放送分数',
                '終了時刻', '大分類コード', '中分類コード',
                '分類コード', '新番・特番フラグ',
                '放送形式', '最終回フラグ',
                '番組名（漢字）', '番組名（カナ）(秒)']
file_names = ['../data/processed_data/tv_program_000{}_part_00.csv'.format(i) for i in range(4)]
df_list = [pd.read_csv(file_name, header=None) for file_name in file_names]
program_df = pd.concat(df_list)
program_df.columns = columns_names
program_df.index = program_df["番組名（漢字）"]
program_to_code = dict(program_df.drop_duplicates(subset="番組名（漢字）")["番組コード"])

f = open('res_week_random.txt')
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
        pid = program2id[program_to_code[p]]
    except:
        print('Cannot find this program')
        continue

    embedding = all_embeddings[pid: pid+1]
    d = cosine_similarity(embedding, all_embeddings)[0]
    d = zip(all_programs, d)
    d = sorted(d, key=lambda x: x[1], reverse=True)
    program_info(program_to_code[p], "", program_df)
    for p in d[1:11]:
        if len(p[0]) < 2:
            continue
        program_info(p[0], p[1], program_df)
