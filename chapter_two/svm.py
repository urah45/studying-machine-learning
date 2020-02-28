import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC



# カーネル法を用いたサポートベクターマシン
# 線形サポートベクターマシンを拡張したもの

# 低次元での線形モデルは非常に制約が強い
X, y = make_blobs(centers=4, random_state=8)
y = y % 2
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# plt.show() # これは線形モデルで分類できない

X_new = np.hstack([X, X[:, 1:]**2])
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
# y == 0 をプロットしてからy == 1をプロット
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
# plt.show()　　これなら分類できる

# カーネルトリック
# 非線形の特徴量データを表現に加えることで、
# 線形モデルがはるかに強力になることがわかったが
# 実際には、どの特徴量を加えたらよいかがわからない
# ここで、カーネルトリックという、高次元空間での
# クラス分類器を学習させる巧妙な数学的トリックがある

# svmは、個々のデータポイントが二つのクラスの
# 決定境界表現するのにどの程度重要かを学習する

# svmでは、パラメータの設定と、データのスケールに敏感であるという問題がある
# 特に、すべての特徴量の変異が同じスケールであることを要求する
# スケール変換が必要になってくる

# パラメータ
# カーネル法を用いたSVMで重要なパラメータは正則化パラメータと
# カーネルの選択と、カーネル固有のパラメータである
# RBFカーネルのパラメータは、ガウシアンカーネルの幅の逆数を表すgammaだけである
# gammaとCは両方ともモデルの複雑さを制御するパラメータで、大きくするとより複雑な
# モデルになる。したがって、二つのパラメータの設定は
# 強く相関するため、同時に調整する必要がある