import pandas as pd
import itertools

# Excelファイルを読み込む
file_path = 'pointmutation_dupdel.xlsx'
df = pd.read_excel(file_path)

# mutation列とgain列のデータをリストとして取得
mutations = df['mutation'].tolist()
gains = df['gain'].tolist()

# gain列のデータから3.719212132を引く
adjusted_gains = [gain - 3.719212132 for gain in gains]

# 組み合わせを生成（最大3つまで）
mutation_combinations = []
scores = []

for r in range(1, min(len(mutations), 3) + 1):
    for combo in itertools.combinations(range(len(mutations)), r):
        combo_mutations = ':'.join([mutations[i] for i in combo])
        combo_score = sum([adjusted_gains[i] for i in combo])
        mutation_combinations.append(combo_mutations)
        scores.append(combo_score)

# 新しいデータフレームを作成
df2 = pd.DataFrame({
    'mutationcombo': mutation_combinations,
    'score': scores
})

# scoreを高い順に並べ替え、上位100個を取得
df2_sorted = df2.sort_values(by='score', ascending=False).head(100)

# 出力ファイル名を指定してCSVファイルを保存
output_file = 'hopeful_mutant.csv'
df2_sorted.to_csv(output_file, index=False)

print(f"CSVファイル '{output_file}' を作成しました。")
