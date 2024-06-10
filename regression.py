import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print("[" + current_time() + "] データを読み込んでいます...")
data = pd.read_csv('gfp_data.csv')
print("[" + current_time() + "] データの読み込みが完了しました。")

# 列名を確認するためにデータフレームの最初の数行を表示
print(data.head())

# 列名が正しいことを確認してから使用する
if 'sequence' in data.columns and 'activity' in data.columns:
    sequences = data['sequence']
    activities = data['activity']
else:
    raise ValueError("CSVファイルに'sequence'または'activity'という列が存在しません。")

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

# 不明な文字を除去
cleaned_sequences = clean_sequences(sequences)

# すべての配列の長さを最大長に合わせる
max_sequence_length = max(cleaned_sequences.str.len())
padded_sequences = pad_sequences(cleaned_sequences, max_sequence_length)

encoded_sequences = encode_sequences(padded_sequences)

print("[" + current_time() + "] データを訓練データとテストデータに分割しています...")
X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, activities, test_size=0.2, random_state=42)
print("[" + current_time() + "] データの分割が完了しました。")

print("[" + current_time() + "] モデルを構築しています...")
model = Sequential([
    Dense(1024, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')  # 出力層は1つのニューロンで、活性化関数は線形
])


print("[" + current_time() + "] 入力次元列は" + str(X_train.shape[1]) + "です...")
print("[" + current_time() + "] 入力次元行は" + str(X_train.shape[0]) + "です...")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
print("[" + current_time() + "] モデルの学習を開始します...")
model.fit(X_train, y_train, epochs=10, batch_size=1024, validation_data=(X_test, y_test))
print("[" + current_time() + "] モデルの学習が完了しました。")

print("[" + current_time() + "] 活性値が高いと予測される配列を生成しています...")
# 高活性予測配列の生成（シンプルな例）
predictions = model.predict(X_test)
sorted_indices = np.argsort(predictions[:, 0])[::-1]  # 高い順にソート
top_100_indices = sorted_indices[:100]

top_sequences = np.array(cleaned_sequences)[top_100_indices]
top_activities = predictions[top_100_indices]

results = pd.DataFrame({'sequence': top_sequences, 'predicted_activity': top_activities[:, 0]})
results.to_csv('predicted_high_activity_sequences.csv', index=False)
print("[" + current_time() + "] 結果をCSVファイルに出力しました。")