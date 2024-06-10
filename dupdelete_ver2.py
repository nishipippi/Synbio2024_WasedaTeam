import pandas as pd

# CSVファイルを読み込む
target_file = 'hopeful_point_mutation.xlsx'  # ファイルパスを適切に指定してください
pointmutation_dupdel = 'pointmutation_dupdel.xlsx'
df = pd.read_excel(target_file)

# B列のデータからアルファベットを取り除いてE列にコピー
df['point'] = df['mutation'].str.replace(r'[a-zA-Z]', '', regex=True)

# ポイントを分割してリスト化
df['point_list'] = df['point'].str.split(':')

# 重複を検出して行を削除する
unique_points = set()
rows_to_keep = []

for index, row in df.iterrows():
    points = row['point_list']
    if not any(point in unique_points for point in points):
        rows_to_keep.append(index)
        unique_points.update(points)

df_filtered = df.loc[rows_to_keep]

# 不要な列を削除
df_filtered = df_filtered.drop(columns=['point_list'])

# 結果を新しいxlsxファイルに保存
df_filtered.to_excel(pointmutation_dupdel, index=False)

print(f"処理が完了しました。結果は{pointmutation_dupdel}に保存されています。")
