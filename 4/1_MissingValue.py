import pandas as pd
from io import StringIO
# サンプルデータを作成
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))

'''
print('sample\n',df)
# 各特徴量の欠損値をカウント
print('countNA\n',df.isnull().sum())
# 欠損値を含んでいる行を削除
print('行削除\n', df.dropna(axis=0))
# 欠損値を含んでいる列を削除
print('列削除\n', df.dropna(axis=1))
# 全ての列がNaNである行だけを削除
print('allNaN削除\n',df.dropna(how='all'))
# 非NaN値が4つ未満の行を削除
print('非NaN4未満\n',df.dropna(thresh=4))
# 特例の列(この場合は'C')にNaNが含まれている行だけを削除
print('特定列削除\n',df.dropna(subset=['C']))
'''

from sklearn.impute import SimpleImputer
import numpy as np
# 欠損値補完のインスタンスを生成（平均値補完）
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
# データに適合させる
imr = imr.fit(df.values)
# 補完を実行
imputed_data = imr.transform(df.values)
# print(imputed_data)
# print(df.fillna(df.mean()))

