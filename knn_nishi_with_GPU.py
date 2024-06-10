import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import time

def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print("[" + current_time() + "] gfp_data.csvデータを読み込んでいます...")
gfp_data = pd.read_csv('gfp_data.csv')
print("[" + current_time() + "] gfp_data.csvデータの読み込みが完了しました。")

print("[" + current_time() + "] mutant_output.csvデータを読み込んでいます...")
mutant_data = pd.read_csv('mutant_output.csv')
print("[" + current_time() + "] mutant_output.csvデータの読み込みが完了しました。")

# 列名を確認するためにデータフレームの最初の数行を表示
print(gfp_data.head())
print(mutant_data.head())

# gfp_data.csvの列名が正しいことを確認してから使用する
if 'created sequence' in gfp_data.columns and 'gain' in gfp_data.columns:
    gfp_sequences = gfp_data['created sequence']
    gfp_activities = gfp_data['gain']
else:
    raise ValueError("gfp_data.csvファイルに'created sequence'または'gain'という列が存在しません。")

# mutant_output.csvの列名が正しいことを確認してから使用する
if 'created sequence' in mutant_data.columns:
    mutant_sequences = mutant_data['created sequence']
else:
    raise ValueError("mutant_output.csvファイルに'created sequence'という列が存在しません。")

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

def encode_sequences(sequences):
    print("[" + current_time() + "] アミノ酸配列をエンコードしています...")
    amino_acids = list('ACDEFGHIKLMNPQRSTVWYX')  # 'X' for padding
    label_encoder = LabelEncoder()
    label_encoder.fit(amino_acids)
    integer_encoded = [label_encoder.transform(list(seq)) for seq in sequences]
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    integer_encoded = np.array(integer_encoded)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print("[" + current_time() + "] エンコードが完了しました。")
    return onehot_encoded

# gfp_data.csvの不明な文字を除去
cleaned_gfp_sequences = clean_sequences(gfp_sequences)

# mutant_output.csvの不明な文字を除去
cleaned_mutant_sequences = clean_sequences(mutant_sequences)

# すべての配列の長さを最大長に合わせる
max_sequence_length = max(max(cleaned_gfp_sequences.str.len()), max(cleaned_mutant_sequences.str.len()))
padded_gfp_sequences = pad_sequences(cleaned_gfp_sequences, max_sequence_length)
padded_mutant_sequences = pad_sequences(cleaned_mutant_sequences, max_sequence_length)


# 配列を結合してエンコード
combined_sequences = padded_gfp_sequences + padded_mutant_sequences
encoded_sequences = encode_sequences(combined_sequences)

# エンコードされたデータを分割
encoded_gfp_sequences = encoded_sequences[:len(padded_gfp_sequences)]
encoded_mutant_sequences = encoded_sequences[len(padded_gfp_sequences):]


# CUDAが利用可能か確認
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'
    print("[" + current_time() + "] GPUが利用可能です。")
else:
    device = '/CPU:0'
    print("[" + current_time() + "] GPUが利用できないため、CPUを使用します。")

with tf.device(device):
    print("[" + current_time() + "] モデルを構築しています...")
    model = KNeighborsRegressor(n_neighbors=5)

    print("[" + current_time() + "] モデルの学習を開始します...")
    model.fit(encoded_gfp_sequences, gfp_activities)
    print("[" + current_time() + "] モデルの学習が完了しました。")

    print("[" + current_time() + "] mutant_output.csvデータに対して活性値を予測しています...")
    mutant_predictions = model.predict(encoded_mutant_sequences)

results = pd.DataFrame({'created sequence': mutant_sequences, 'predicted_activity': mutant_predictions})
results.to_csv('predicted_mutant_activities.csv', index=False)
print("[" + current_time() + "] 結果をCSVファイルに出力しました。")
