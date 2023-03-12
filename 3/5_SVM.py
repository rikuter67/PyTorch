from sklearn.svm import SVC
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
        

svm = SVC(kernel='linear', C=1.0, random_state=1) # 線形SVMのインスタンスを生成
svm.fit(X_train_std, y_train)

# 訓練データとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# 訓練データとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))

# 決定領域をプロット
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
# plt.xlabel('Petal length [standardized]') # 軸ラベルを設定
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left') # 凡例を設定
# plt.tight_layout()
# plt.show()

#scikit-learnでの代替実装（SGDClassifier）
#from sklearn.linear_model import SGDClassifier
#ppn = SGDClassifier(loss='perceptron') #SGDバージョンのパーセプトロン
#lr = SGDClassifier(loss='log')         #SGDバージョンのロジスティック回帰
#svm = SGDClassifier(loss='hinge')       #SGDバージョンのSVM（損失関数＝ヒンジ関数）


#-----------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1) #乱数シードを指定
X_xor = np.random.randn(200,2) #標準正規分布に従う乱数で200行2列の行列を作成
# 2つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# 排他的論理和の値が芯の場合は１，偽の場合は０を割り当てる
y_xor = np.where(y_xor, 1, 0)
# # ラベル１を青の四角でプロット
# plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='royalblue', marker='s', label='Class 1')
# # ラベル０を赤の円でプロット
# plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c='tomato', marker='o', label='Class 0')
# # 軸の範囲を設定
# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()


# # RBFカーネルによるSVMのインスタンスを生成
# svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
# svm.fit(X_xor, y_xor)
# plot_decision_regions(X_xor, y_xor, classifier=svm)
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()


# RBFカーネルによるSVMのインスタンスを生成（2つのパラメータを変更）
# svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
# svm.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
# plt.xlabel('Petal length [standardized]')
# plt.ylabel('Petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# plt.show()

# RBFカーネルによるSVMのインスタンスを生成（γパラメータを変更）
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()