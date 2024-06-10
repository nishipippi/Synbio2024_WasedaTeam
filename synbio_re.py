import pandas as pd

# CSVファイルの読み込み
input_file = 'hopeful_mutant.csv'
output_file = 'mutant_output.csv'

# データを読み込む
df = pd.read_csv(input_file)

# original_sequence_column と substitutions_column を適切な列名に変更
original_sequence_column = 'original'  # A列の名前
substitutions_column = 'mutation'      # B列の名前

# E列を準備（E列の初期値としてA列の配列をコピー）
df['created sequence'] = df[original_sequence_column]  # E列に置換後の配列を格納するための列を追加

# 置換を適用する関数
def apply_substitution(sequence, substitution):
    original_aa = substitution[0]  # 置換前のアミノ酸
    position = int(substitution[1:-1])  # mutation列の数字は0ベース
    new_aa = substitution[-1]  # 置換後のアミノ酸

    # 元の配列の長さを超えないように条件を追加
    if position >= len(sequence):
        return sequence

    # '*' が出現した場合、その残基を欠損として扱う
    if new_aa == '*':
        modified_sequence = sequence[:position] + '-' + sequence[position+1:]
    else:
        # 置換を適用
        if sequence[position] == original_aa:
            modified_sequence = sequence[:position] + new_aa + sequence[position+1:]
        else:
            modified_sequence = sequence  # 置換が適用できない場合はそのまま

    return modified_sequence

# 各行に対して置換を適用
for index, row in df.iterrows():
    original_sequence = row[original_sequence_column]
    substitution_info = row[substitutions_column]

    if substitution_info == 'WT':
        modified_sequence = original_sequence
    else:
        substitutions = substitution_info.split(':')
        modified_sequence = original_sequence
        for substitution in substitutions:
            modified_sequence = apply_substitution(modified_sequence, substitution)

    df.at[index, 'created sequence'] = modified_sequence

# difference列を作成して変化部分を記録
df['difference'] = [''.join(['*' if e != o else ' ' for e, o in zip(row['created sequence'], row['original'])]) for index, row in df.iterrows()]

# 結果を新しいCSVファイルに書き込む
df.to_csv(output_file, index=False)

print("処理が完了しました。結果は 'mutant_output.csv' に保存されました。")
