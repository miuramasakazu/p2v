import pandas as pd
from collections import defaultdict

watch_program_df = pd.read_pickle("watch_program_df.pkl")
watch_group = watch_program_df.groupby(["世帯No", "番組コード"])
watch_ratio = watch_group[["視聴分数"]].sum()

columns_names = ['局コード', '番組コード', '曜日コード',
                '放送日', '開始時刻', '放送分数',
                '終了時刻', '大分類コード', '中分類コード',
                '分類コード', '新番・特番フラグ',
                '放送形式', '最終回フラグ',
                '番組名（漢字）', '番組名（カナ）(秒)']
file_names = ['data/processed_data/tv_program_000{}_part_00.csv'.format(i) for i in range(4)]
df_list = [pd.read_csv(file_name, header=None) for file_name in file_names]
program_df = pd.concat(df_list)
program_df.columns = columns_names

def ratio(d, program_time):
    return program_time[d["番組コード"]]
program_time = program_df.groupby(["番組コード"])["放送分数"].sum()
watch_ratio = pd.merge(watch_ratio, program_time.to_frame(), left_index=True, right_index=True)
watch_ratio["視聴割合"] = watch_ratio["視聴分数"] / watch_ratio["放送分数"]

P = 0.01
watch_dict = defaultdict(list)
d = list(watch_ratio.query('視聴割合 > @P').index)
for (sample, code) in d:
    watch_dict[sample].append(code)

sequence = []
for sample in watch_dict.values():
    sequence.append(' '.join(sample))

s = pd.Series(sequence)
s.to_csv('train.txt', sep='\t', index=False)
