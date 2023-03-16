import pandas as pd
# サンプルデータを生成（Tシャツの色・サイズ・価格・クラスラベル）
df = pd.DataFrame([
    ['green', 'M', '10.1', 'class2'],
    ['red', 'L', '13.5', 'class1'],
    ['blue', 'XL', '15.3', 'class2']])
# 列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
# print(df)

# 後から整数値を元の文字列表現に戻したい場合
inv_size_mapping = {v: k for k, v in size_mapping.items()}
# print(df['size'].map(inv_size_mapping))

import numpy as np
# クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# print(class_mapping)

# クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
# print(df)

# 後からもとの文字列表現に戻したい場合
# 整数とクラスラベルを対応させるディクショナリを生成
inv_calss_mapping = {v: k for k, v in class_mapping.items()}
# 整数からクラスラベルに変換
df['classlabel'] = df['classlabel'].map(inv_calss_mapping)
# print(df)

# scikit-learnを用いる場合
from sklearn.preprocessing import LabelEncoder
# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
# クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
# print(y)

# クラスラベルを文字列に戻す
# print(class_le.inverse_transform(y))

X = df[['color', 'size', 'price']].values #Tシャツの色、サイズ、価格を抽出
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
# print(X)

from sklearn.preprocessing import OneHotEncoder
X = df[['color', 'size', 'price']].values
# one-hot エンコーダを生成
color_ohe = OneHotEncoder()
# one-hotエンコーディングを実行
# print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())

from sklearn.compose import ColumnTransformer
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]),
                              ('nothing', 'passthrough', [1, 2])])
c_transf.fit_transform(X).astype(float)

# one-hotエンコーディングを実行
# print(pd.get_dummies(df[['price', 'color', 'size']]))

# one-hotエンコーディングを実行
# print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))

# one-hot エンコーダの生成
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([('onehot', color_ohe, [0]),
                              ('nothing', 'passthrough', [1, 2])])
# print(c_transf.fit_transform(X).astype(float))

# 順序特徴量のエンコーディング
df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', '13.5', 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']
# print(df)

df['x > M'] = df['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(lambda x: 1 if x == 'XL' else 0)
del df['size']
print(df)