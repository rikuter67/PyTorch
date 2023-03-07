import matplotlib.pyplot as plt
import numpy as np
# シグモイド関数を定義
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# # 0.1間隔で-7以上7未満のデータを生成
# z = np.arange(-7, 7, 0.1)
# # 生成したデータでシグモイド関数を実行
# sigma_z = sigmoid(z)
# # 元のデータとシグモイド関数の出力をプロット
# plt.plot(z, sigma_z)
# # 垂直線を追加(z=0)
# plt.axvline(0.0, color='k')
# # y軸の上限/下限を設定
# plt.ylim(-0.1, 1.1)
# # 軸ラベルを設定
# plt.xlabel('z')
# plt.ylabel('$\sigma (z)$')
# # y軸の目盛を追加
# plt.yticks([0.0, 0.5, 1.0])
# # Axesクラスのオブジェクトの所得
# ax = plt.gca()
# # y軸の目盛に合わせて水平グリット線を追加
# ax.yaxis.grid(True)
# # グラフを表示
# plt.tight_layout()
# plt.show()

# y=1の損失値を計算する関数
def loss_1(z):
    return -np.log(sigmoid(z))

# y=0の損失値を計算する関数
def loss_0(z):
    return -np.log(1 - sigmoid(z))

# 0.1間隔で-10以上10未満のデータを生成
z = np.arange(-10, 10, 0.1)
#シグモイド関数を実行
sigma_z = sigmoid(z)
# y=1の損失値を計算する関数を実行
c1 = [loss_1(x) for x in z]
# 結果をプロット
plt.plot(sigma_z, c1, label='L(w,b) if y=1')
# y=0の損失値を計算する関数を実行
c0 = [loss_0(x) for x in z]
# 結果をプロット
plt.plot(sigma_z, c0, linestyle='--', label='L(w,b) if y=0')
# x軸とy軸の上限/下限を設定
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
# 軸のラベルを設定
plt.xlabel('$\sigma(z)')
plt.ylabel('L(w, b)')
# 凡例を設定
plt.legend(loc='best')
# グラフ表示
plt.tight_layout()
plt.show()