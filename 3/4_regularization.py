from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Irisデータセットをロード
iris = datasets.load_iris()
# 3,4列目の特徴量を抽出
X = iris.data[:, [2, 3]]
# クラスらラベルを取得
y = iris.target
# 一意なクラスラベルを出力
# print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
# 訓練データとテストデータに分割：全体の30％をテストデータにする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 訓練データの平均と標準偏差を計算
sc.fit(X_train)
# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # マーカーとカラーマップの準備
    markers = ('o','s','^','v','<')
    colors = ('red','blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # 各特徴量を1次元配列に変換して予測を実行
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    lab = lab.reshape(xx1.shape)
    # グリッドポイントの等高線をプロット
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap) #軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    # テストデータ点を目立たせる（点を◯で表示）
    if test_idx:
        # すべてのデータ点プロット
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black',
                    alpha=1.0, linewidth=1, marker='o', s=100, label='Test set')
        

# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
# 訓練データをモデルに適合させる
lr.fit(X_train_std, y_train)

# 訓練データとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# 訓練データとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))

# # 決定領域をプロット
# plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
# plt.xlabel('Petal length [standardized]') # 軸ラベルを設定
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left') # 凡例を設定
# plt.tight_layout()
# plt.show()

# print(lr.predict_proba(X_test_std[:3,:]))
# print(lr.predict_proba(X_test_std[:3,:]).argmax(axis=1))
# print(lr.predict(X_test_std[:3,:]))
# print(lr.predict(X_test_std[0,:].reshape(1,-1)))

weights, params = [], [] # 重み係数と逆正則化パラメータのリストを生成
# 10個の逆正則化パラメータに対応するロジスティック回帰モデルを処理
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1]) # 重み係数を格納
    params.append(10.**c) # 逆正則化パラメータを格納

weights = np.array(weights) #重み係数をNumpy配列に変換
# 横軸に逆正則化パラメータ、縦軸に重み係数をプロット
plt.plot(params, weights[:, 0], label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='Petal width')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log') #横軸を対数スケールに設定
plt.show()