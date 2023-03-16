import numpy as np
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total Phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

from sklearn.preprocessing import MinMaxScaler
# min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()
# 訓練データをスケーリング
X_train_norm = mms.fit_transform(X_train)
# テストデータをスケーリング
X_test_norm = mms.transform(X_test)

# 標準化とmin-maxスケーリング（正規化）の比較
ex = np.array([0, 1, 2, 3, 4, 5])
# print('standardized:', (ex - ex.mean()) / ex.std())
# print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

# scikit-learnの場合
from sklearn.preprocessing import StandardScaler
# 標準化のインスタンスを生成（平均＝０、標準偏差＝１に変換）
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print(X_train_std)