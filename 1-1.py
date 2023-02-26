import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pdb

class Perceptron:
    """パーセプトロン分類器

    パラメータ
    -----------------
    eta : float
        学習率(0.0より大きく1.0以下の値)
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    ----------------
    w_ : 1次元配列
        適合後の重み
    b_ : スカラー
        適用後のバイアスユニット
    errors_ : リスト
        各エポックでの誤分類(更新)の数
    
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """訓練データに適合させる

        パラメータ
        --------------
        X : {配列のようなデータ構造}, shape  [n_examples, n_features]
            訓練ベクトル : n_examplesは訓練データの個数、n_featureは特徴量の個数
        y : 配列のようなデータ構造, shape = [n_examples]
            目的変数
        
        戻り値
        --------------
        self : object

        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)

        self.errors_ = []
        
        for _ in range(self.n_iter): #訓練データを繰り返し処理
            errors = 0
            for xi, target in zip(X, y): #重みとバイアスユニットを更新
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0) #誤差を追加
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions(X, y, classifier, resolution=0.02):

    #マーカーとカラーマップの準備
    markers = ('o','s','^','v','<')
    colors = ('red','blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    #各特徴量を1次元配列に変換して予測を実行
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #予測結果を元のグリッドポイントのデータサイズに変換
    lab = lab.reshape(xx1.shape)
    #グリッドポイントの等高線をプロット
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap) #軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

if __name__ == '__main__':
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(s, header=None, encoding='utf-8')
    
    y = df.iloc[0:100, 4].values #1-100行目の目的変数の抽出
    y = np.where(y == 'Iris-setosa', 0, 1) #Iris-setosaを0, Iris-versicolorを1に変換
    X = df.iloc[0:100, [0,2]].values #1-100行目の1,3列目の抽出
    # plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='Setosa')
    # plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='s', label='Versicolor')
    # #軸ラベルの設定
    # plt.xlabel('Sepal length [cm]')
    # plt.ylabel('Petal length [cm]')
    # #凡例の設定
    # plt.legend(loc='upper left')
    # plt.show()
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # #軸ラベルの設定
    # plt.xlabel('Epoch')
    # plt.ylabel('Number of updates')
    # plt.show()

    #決定領域のプロット
    plot_decision_regions(X, y, classifier=ppn)
    #軸ラベルの設定
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    #凡例の設定(左上に配置)
    plt.legend(loc='upper left')
    plt.show()
