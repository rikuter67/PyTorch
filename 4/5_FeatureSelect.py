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

# scikit-learnの場合
from sklearn.preprocessing import StandardScaler
# 標準化のインスタンスを生成（平均＝０、標準偏差＝１に変換）
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

from sklearn.linear_model import LogisticRegression
# L1正則化ロジスティック回帰のインスタンスを生：逆正則化パラメータC=1.0はデフォルト値であり、
# 値を大きくしたり小さくしたりすると、正則化の効果を強めたり弱めたりできる
lr = LogisticRegression(penalty='l1', solver='liblinear', multi_class='ovr')
# 訓練データに適合
lr.fit(X_train_std, y_train)
# 訓練データに対する正解率の表示
# print('Training accuracy:', lr.score(X_train_std, y_train))
# print('Test accuracy:', lr.score(X_test_std, y_test))

# print(lr.intercept_)
# print(lr.coef_)


import matplotlib.pyplot as plt
# # 描画の準備
# fig = plt.figure()
# ax = plt.subplot(111)
# # 各係数の色のリスト
# colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
# # 空のリストを生成（重み係数、逆正則化パラメータ）
# weights, params = [], []
# # 逆正則化パラメータの値ごとの処理
# for c in np.arange(-4., 6.):
#     lr = LogisticRegression(penalty='l1', C=10.**c, solver='liblinear', multi_class='ovr', random_state=0)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10**c)

# # 重み係数をNumpy配列に変換
# weights = np.array(weights)
# # 各重み係数をプロット
# for column, color in zip(range(weights.shape[1]), colors):
#     # 横軸を逆正則化パラメータ、縦軸を重み係数とした折れ線グラフ
#     plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)
# # y=0に黒い波線を引く
# plt.axhline(0, color='black', linestyle='--', linewidth=3)
# # 横軸の範囲の設定
# plt.xlim([10**(-5), 10**5])
# # 軸のラベルの設定
# plt.ylabel('Weight coefficient')
# plt.xlabel('C (inverse regularization strength)')
# # 横軸を対数スケールに設定
# plt.xscale('log')
# plt.legend(loc='upper left')
# ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
# plt.show()


from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS:
    """
    逐次後退選択(sequential backward selection)を実行するクラス
    """

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring              # 特徴量を評価する指標
        self.estimator = clone(estimator)   # 推定器
        self.k_features = k_features        # 選択する特徴量の個数
        self.test_size = test_size          # テストデータの割合
        self.random_state = random_state    # 乱数シードを固定する random_state

    def fit(self, X, y):
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        # 全ての特徴量の個数、列インデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        # 全ての特徴量を用いてスコアを算出
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]       # スコアを格納
        # 特徴量が指定した個数になるまで処理を繰り返す
        while dim > self.k_features:
            scores = []               # 空のスコアリストを作成
            subsets = []             # 空の列インデックスリストを作成
            # 特徴量の部分集合を表す列インデックスの組み合わせごとに処理を反復
            for p in combinations(self.indices_, r=dim-1):
                # スコアを算出して格納
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                # 特徴量の部分集合を表す列インデックスのリストを格納
                subsets.append(p)

            # 最良のスコアのインデックスを抽出
            best = np.argmax(scores)
            # 最良のスコアとなる列インデックスを抽出して格納
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            # 特徴量の個数を1つだけ減らして次のステップへ
            dim -= 1
            # スコアを格納
            self.scores_.append(scores[best])

        # 最後に格納したスコア
        self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):
        # 抽出した特徴量を返す
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列番号indicesの特徴量を抽出してモデルを適合
        self.estimator.fit(X_train[:, indices], y_train)
        # テストデータを用いてクラスラベルを予測
        y_pred = self.estimator.predict(X_test[:, indices])
        # クラスラベルの正解値と予測値を用いてスコアを算出
        score = self.scoring(y_test, y_pred)
        return score
    
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# k最近傍法分類器のインスタンスを生成（近傍点数＝５）
knn = KNeighborsClassifier(n_neighbors=5)
# 逐次後退選択のインスタンスを生成（特徴量の個数が1になるまで特徴量を選択）
sbs = SBS(knn, k_features=1)
# 逐次後退選択を実行
sbs.fit(X_train_std, y_train)

# 特徴量の個数リスト（13,12,......,1）
k_feat = [len(k) for k in sbs.subsets_]
# 横軸を特徴量の個数、縦軸をスコアとした折れ線グラフのプロット
# plt.plot(k_feat, sbs.scores_, marker='o')
# plt.ylim([0.7, 1.02])
# plt.ylabel('Accuracy')
# plt.xlabel('Number of features')
# plt.grid()
# plt.tight_layout()
# plt.show()

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

# 13個全ての特徴量を使ってモデルを適合させる
knn.fit(X_train_std, y_train)
# 訓練の正解率を出力
print('Training accuracy:', knn.score(X_train_std, y_train))
# テストの正解率を出力
print('Test accuracy:', knn.score(X_test_std, y_test))

# 3つの特徴量を用いてモデルを適合
knn.fit(X_train_std[:, k3], y_train)
# 訓練の正解率を出力
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))