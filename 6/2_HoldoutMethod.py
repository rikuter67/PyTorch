import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/wdbc.data', header=None)

from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
# 連結する処理としてスケーリング、主成分分析、ロジスティック回帰を指定
pipe_lr = make_pipeline(StandardScaler(),
                       PCA(n_components = 2),
                       LogisticRegression())

# 分割元のデータ、分割数、乱数生成器の状態を指定し、
# 層化k分割交差検証イテレータを表すStratifiedKFoldクラスをインスタンス化
kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []
# イテレータのインデックスと要素をループ処理：（上から順に）
#       データをモデルに適合
#       テストデータの正解率を算出
#       リストに正解率を追加
#       分割の番号、0 以上の要素数、正解率を出力
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f'Fold: {k+1:02d}, '
          f'Class distr.: {np.bincount(y_train[train])},'
          f'Acc.: {score:.3f}')
    
# 正解率の平均と標準偏差を出力
mean_acc = np.mean(scores)
std_acc = np.std(scores)
print(f'\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}')

from sklearn.model_selection import cross_val_score
# 交差検証のcross_val_score関数でモデルの正解率を算出
# 推定器estimator、訓練データX、予測値y、分割数cv、CPU数n_jobsを指定
scores = cross_val_score(estimator=pipe_lr,
                         X=X_train, y=y_train,
                         cv=10, n_jobs=1)
print(f'CV accuracy scores: {scores}')

print(f'CV accuracy: {np.mean(scores):.3f}'
      f' +/- {np.std(scores):.3f}')