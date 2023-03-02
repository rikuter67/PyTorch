import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

class AdalineGD:
    """ADAptive LInear NEuron分類器
    
    パラメータ
    --------------
    eta : float
        学習率(0.0より大きく1.0以下の値)
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを書記かするための乱数シード
        
    属性
    --------------
    w_ : 1次元の配列
        適合後の重み
    b_ : スカラー
        適合後のバイアス
    losses_ : リスト
        各エポックでのMSE誤差関数の値
        
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ 訓練データに適合させる

        パラメータ
        --------------
        X : {配列のようなデータ構造}, shape = [n_examples, n_features]
        訓練データ
        y : 配列のようなデータ構造, shape=[n_examples]
        目的変数

        戻り値
        --------------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter): #訓練回数分まで訓練データを反復
            net_input = self.net_input(X)
            #activateメソッドは単なる高等関数であるため、
            #このコードでは何の効果もないことに注意。代わりに、
            #直接'output = self.net_input(X)'と記述することも出来た。
            #activationメソッドの目的は、より概念的なものである。
            #つまり、(後ほど説明する)ロジステック回帰の場合は、
            #ロジスティック回帰の分類器を実装するためにシグモイド関数に変更することも出来る
            output = self.activation(net_input)
            #誤差の計算
            errors = (y - output)
            #重みの更新
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            #バイアスの更新
            self.b_ += self.eta * 2.0 * errors.mean()
            #損失関数の計算
            loss = (errors**2).mean()
            #損失関数の追加
            self.losses_.append(loss)
        return self
    
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

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
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
    # #描画領域を1行2列に分割
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    # #勾配降下法によるADALINEの学習(学習律 eta=0.1)
    # ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X,y)
    # #エポック数と損失関数の関係を表す折れ線グラフのプロット(縦軸の損失関数は常用対数)
    # ax[0].plot(range(1,len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
    # #軸ラベルの設定
    # ax[0].set_xlabel('Epochs')
    # ax[0].set_ylabel('log(Mean squared error)')
    # #タイトルの設定
    # ax[0].set_title('Adaline - Learning rate 0.1')
    # #勾配降下法によるADALINEの学習(学習律 eta=0.0001)
    # ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X,y)
    # #エポック数と損失関数の関係を表す折れ線グラフのプロット
    # ax[1].plot(range(1,len(ada2.losses_) + 1), ada2.losses_, marker='o')
    # #軸ラベルの設定
    # ax[1].set_xlabel('Epochs')
    # ax[1].set_ylabel('log(Mean squared error)')
    # #タイトルの設定
    # ax[1].set_title('Adaline - Learning rate 0.0001')
    # #図の表示
    # plt.show()

    #データのコピー
    X_std = np.copy(X)
    #各列の標準化
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    #勾配降下法によるADALINEの学習(標準化後、学習率eta=0.5)
    ada_gd = AdalineGD(n_iter=20, eta=0.5)
    #モデルの適合
    ada_gd.fit(X_std, y)
    #決定領域のプロット
    plot_decision_regions(X_std, y, classifier=ada_gd)
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
    plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
    #軸ラベルの設定
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    #図の表示
    plt.tight_layout()
    plt.show()