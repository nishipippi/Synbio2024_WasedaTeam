
import tqdm
import numpy as np
import pandas as pd
from itertools import combinations
import time

def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print("[" + current_time() + "] データを読み込んでいます...")
data = pd.read_csv('gfp_data.csv')
print("[" + current_time() + "] データの読み込みが完了しました。")

# 列名を確認するためにデータフレームの最初の数行を表示
print(data.head())

# 列名が正しいことを確認してから使用する
if 'sequence' in data.columns and 'activity' in data.columns and 'kind' in data.columns:
    avGFP_data = data[data['kind'] == 'avGFP']
    sequences = avGFP_data['sequence']
    activities = avGFP_data['activity']
else:
    raise ValueError("CSVファイルに'sequence'、'activity'または'kind'という列が存在しません。")

def clean_sequences(sequences):
    sequences = sequences.str.replace('-', '', regex=False)
    sequences = sequences.str.replace('.', '', regex=False)
    return sequences

def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded_seq = seq[:maxlen]
        else:
            padded_seq = seq + 'X' * (maxlen - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences

def hamming_distance(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length.")
    distance = 0
    for c1, c2 in zip(seq1, seq2):
        if c1 != c2:
            distance += 1
        if distance > 2:
            return distance  # 3以上になった時点で計算を中断
    return distance

# 与えられた配列とそのactivity
given_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
given_activity = 3.719212132

# 不明な文字を除去
cleaned_sequences = clean_sequences(sequences)

# すべての配列の長さを最大長に合わせる
max_sequence_length = max(cleaned_sequences.str.len())
padded_sequences = pad_sequences(cleaned_sequences, max_sequence_length)

# 与えられた配列もパディング
padded_given_sequence = pad_sequences([given_sequence], max_sequence_length)[0]

# ハミング距離が1および2の組み合わせを抽出
combinations_hamming_1 = []
combinations_hamming_2 = []

for seq, activity in tqdm.tqdm(zip(padded_sequences, activities)):
    distance = hamming_distance(padded_given_sequence, seq)
    if distance == 1:
        difference = given_activity - activity
        combinations_hamming_1.append((given_sequence, seq, given_activity, activity, difference))
    elif distance == 2:
        difference = given_activity - activity
        combinations_hamming_2.append((given_sequence, seq, given_activity, activity, difference))

# 結果をデータフレームに保存
df_hamming_1 = pd.DataFrame(combinations_hamming_1, columns=['given_sequence', 'sequence', 'given_activity', 'activity', 'difference'])
df_hamming_2 = pd.DataFrame(combinations_hamming_2, columns=['given_sequence', 'sequence', 'given_activity', 'activity', 'difference'])

# CSVファイルに保存
df_hamming_1.to_csv('hamming_distance_1_combinations.csv', index=False)
df_hamming_2.to_csv('hamming_distance_2_combinations.csv', index=False)

print("[" + current_time() + "] ハミング距離が1および2の組み合わせをCSVファイルに保存しました。")
