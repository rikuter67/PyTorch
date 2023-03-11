from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


class LogisticRegressionGD:
    """勾配降下法に基づくロジスティック回帰分類器
    
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
    losses_ : リスト
        各エポックでのMSE損失関数の値

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        #学習率の初期化、訓練回数の初期化、乱数シードを固定にするrandom_state
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
        self : LogisticRegressionGDのインスタンス
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        #訓練回数分まで訓練データを反復処理
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1-output))) / X.shape[0])
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_) + self.b_
    def activation(self, z):
        """ロジスティックシグモイド活性化関数を計算"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
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

    x_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    # ロジスティック回帰のインスタンスを生成
    lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
    # モデルを訓練データに適合させる
    lrgd.fit(x_train_01_subset, y_train_01_subset)
    #決定領域をプロット
    plot_decision_regions(X=x_train_01_subset, y=y_train_01_subset, classifier=lrgd)
    plt.xlabel('Petal length [standardized]')
    plt.ylabel('Petal width [standardized]')
    plt.tight_layout()
    plt.show()