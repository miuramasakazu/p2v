import pandas as pd
import datetime

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
program_groupby = program_df.groupby(["局コード", "放送日"])

column_names = ["放送日", "曜日", "世帯No", "個人No",
                "開始時刻", "終了時刻","局コード",
                "TVNo", "データ集計区分", "データSEQ"]
file_names = [f'data/processed_data/tv_orgn_p_cv_000{i}_part_00.csv' for i in range(4)]
df_list = [pd.read_csv(file_name, header=None) for file_name in file_names]
watch_df = pd.concat(df_list)
watch_df.columns = column_names
watch_df = watch_df.query('データ集計区分 == "1"')
watch_df = watch_df.sort_values(["放送日", "世帯No", "開始時刻"])

def watch_time(s, start, end):
    return  pd.Series([max(start, s["開始時刻"]), min(end, s["終了時刻"])])

def to_datetime(data):
    day, time= data[0], data[1]
    if len(day) == 10:
        date = datetime.datetime.strptime(day, '%Y-%m-%d')
    elif len(day) == 8:
        date = datetime.datetime.strptime(day, '%y-%m-%d')
    else:
        raise ValueError("日付の入力エラー")
    hours = time // 100
    minutes = time % 100
    timedelta = datetime.timedelta(hours=hours, minutes=minutes)
    return date + timedelta

def to_minutes(d):
    return (d[1] - d[0]).total_seconds() // 60

def watch_program(watch_data, program_groupby):
    date, code = watch_data["放送日"], watch_data["局コード"]
    start, end = watch_data["開始時刻"], watch_data["終了時刻"]
    df = program_groupby.get_group((code, date)).query('@start <= 終了時刻 and 開始時刻 <= @end')
    df[["視聴開始時刻", "視聴終了時刻"]] = df.apply(watch_time, args=(start, end), axis=1)
    df["番組開始dt"] = df[["放送日", "開始時刻"]].apply(to_datetime, axis=1)
    df["番組終了dt"] = df[["放送日", "終了時刻"]].apply(to_datetime, axis=1)
    df["視聴開始dt"] = df[["放送日", "視聴開始時刻"]].apply(to_datetime, axis=1)
    df["視聴終了dt"] = df[["放送日", "視聴終了時刻"]].apply(to_datetime, axis=1)
    df["視聴分数"] = df[["視聴開始dt", "視聴終了dt"]].apply(to_minutes, axis=1)
    df["世帯No"] = watch_data["世帯No"]
    return df

df = watch_df.apply(watch_program, args=(program_groupby, ), axis=1)
watch_program_df = pd.concat([df.iloc[i] for i in range(len(df))])
watch_program_df.to_pickle('watch_program_df.pkl')
 
