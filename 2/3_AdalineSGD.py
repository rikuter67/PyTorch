import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd


class AdalineSGD:
    """ADAptive LInear NEuron 分類器
    
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

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta                      #学習率の初期化
        self.n_iter = n_iter                #訓練回数の初期化
        self.w_initialized = False          #重みの初期化フラグはFalseに設定
        self.shuffle = shuffle              #各エポックで訓練データをシャッフルするかどうか
        self.random_state = random_state    #乱数シードを設定

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

        self._initialize_weights(X.shape[1])    #重みベクトルの生成
        self.losses_ = []                       #損失値を格納するリストを生成
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            #各訓練データの損失値を格納するリストを生成
            losses = []
            for xi, target in zip(X, y):
                #特徴量xiと目的変数yを使った重みの更新と損失値の計算
                losses.append(self._update_weights(xi, target))
            #訓練データの平均損失値の計算
            avg_loss = np.mean(losses)
            #平均損失値を格納
            self.losses_.append(avg_loss)
        return self
    
    def partial_fit(self, X, y):
        """重みを再初期化することなく訓練データに適合させる"""
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._intialize_wights(X.shape[1])
        # 目的変数yの要素数が2以上の場合は各訓練データの特徴量xiと目的変数targetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        # 目的変数yの要素数が1の場合は訓練データ全体の特徴量Xと目的変数yで重みを更新
        else:
            self._update_weights(X, y)
        return self
    
    def _shuffle(self, X, y):
        """訓練データをシャッフル"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """重みを小さな乱数で初期化"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ADALINEの学習規則を使って重みを更新"""
        output = self.activation(self.net_input(xi))    #活性化関数の出力を計算
        error = (target - output)                       #誤差を計算
        self.w_ += self.eta * 2.0 * xi * (error)        #重みを更新
        self.b_ += self.eta * 2.0 * error               #バイアスを更新
        loss = error**2
        return loss
    
    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return X
    
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
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(s, header=None, encoding='utf-8')
    
    y = df.iloc[0:100, 4].values #1-100行目の目的変数の抽出
    y = np.where(y == 'Iris-setosa', 0, 1) #Iris-setosaを0, Iris-versicolorを1に変換
    X = df.iloc[0:100, [0,2]].values #1-100行目の1,3列目の抽出

    #データのコピー
    X_std = np.copy(X)
    #各列の標準化
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

    # 確率的勾配降下法によるADALINEの学習
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    # モデルへの適合
    ada_sgd.fit(X_std, y)
    # 決定領域のプロット
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    #タイトルの設定
    plt.title('Adaline - Gradient descent')
    #軸ラベルの設定
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    #凡例の設定(左上に配置)
    plt.legend(loc='upper left')
    #図の表示
    plt.tight_layout()
    plt.show()
    #エポック数と損失(MSE)の関係を表す折れ線グラフのプロット
    plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
    #軸ラベルの設定
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    #図の表示
    plt.tight_layout()
    plt.show()