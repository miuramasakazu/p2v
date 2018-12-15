import pandas as pd
import datetime
from tqdm import tqdm
import pandasql as ps

columns_names = ['番組局コード', '番組コード', '曜日コード',
                '番組放送日', '放送開始時刻', '放送分数',
                '放送終了時刻', '大分類コード', '中分類コード',
                '分類コード', '新番・特番フラグ',
                '放送形式', '最終回フラグ',
                '番組名（漢字）', '番組名（カナ）(秒)']
file_names = ['data/processed_data/tv_program_000{}_part_00.csv'.format(i) for i in range(4)]
df_list = [pd.read_csv(file_name, header=None) for file_name in file_names]
program_df = pd.concat(df_list)
program_df.columns = columns_names
program_df = program_df.sort_values(["番組放送日", "番組局コード",  "放送開始時刻"])

column_names = ["放送日", "曜日", "世帯No", "個人No",
                "視聴開始時刻", "視聴終了時刻","局コード",
                "TVNo", "データ集計区分", "データSEQ"]
file_names = [f'data/processed_data/tv_orgn_p_cv_000{i}_part_00.csv' for i in range(4)]
df_list = [pd.read_csv(file_name, header=None) for file_name in file_names]
watch_df = pd.concat(df_list)
watch_df.columns = column_names
watch_df = watch_df.query('データ集計区分 == "1"')
watch_df = watch_df.sort_values(["放送日", "局コード", "視聴開始時刻"])

def to_minutes(d):
    start, end = d[0], d[1]
    hours = end // 100 - start // 100
    minutes = end % 100 - start % 100
    return hours * 60 + minutes

import time
start = time.time()
sqlcode = '''
    select *
    from watch_df
    inner join program_df on watch_df.放送日=program_df.番組放送日 and  watch_df.局コード=program_df.番組局コード
    where watch_df.視聴開始時刻 <= program_df.放送終了時刻 and program_df.放送開始時刻 <= watch_df.視聴終了時刻
'''

watch_program_df = ps.sqldf(sqlcode, locals())
watch_program_df["実視聴開始時刻"] = watch_program_df[["視聴開始時刻", "放送開始時刻"]].max(axis=1)
watch_program_df["実視聴終了時刻"] = watch_program_df[["視聴終了時刻", "放送終了時刻"]].min(axis=1)
watch_program_df["視聴分数"] = (watch_program_df['実視聴終了時刻'] // 100 - watch_program_df['実視聴開始時刻'] // 100) * 60 + \
 (watch_program_df['実視聴終了時刻'] % 100- watch_program_df['実視聴開始時刻'] % 100)
watch_program_df = watch_program_df.drop(['番組放送日', '曜日コード', '番組放送日', '番組名（カナ）(秒)', '番組名（漢字）'], axis=1)
watch_program_df.to_csv('watch_program_df.csv', chunksize=100000)
