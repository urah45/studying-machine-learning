# 線形モデル

import numpy as np
import matplotlib.pyplot as plt

# 線形回帰（通常最小二乗法）
# 訓練データにおいて、予測と真の回帰ターゲットとの
# 平均二乗誤差が最小になるようにｗとｂを求める

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
print(lr.coef_)        # 重み ｗ
print(lr.intercept_)   # 切片 b
# 訓練データから得られた属性は、sklearnでは
# 最後にアンダースコアをつける週間になっている
print(lr.score(X_train, y_train))   # 0.67
print(lr.score(X_test, y_test))  # 0.66
# 適合不足になる

# より複雑なデータセットではどうなるか
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_train, y_train))   # 0.95
print(lr.score(X_test, y_test))     # 0.61
# 過剰適合になる


# リッジ回帰
# L2正則化を行う
# いくつかの係数ｗを0に近づける
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print(ridge.score(X_train, y_train))  # 0.89
print(ridge.score(X_test, y_test))    # 0.75


# ラッソ回帰
# L1正則化を行う
# いくつかの係数ｗを完全に0にする
# モデルが解釈しやすくなり、どの特徴量が重要かが明らかになる
from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print(lasso.score(X_train, y_train))   # 0.29
print(lasso.score(X_test, y_test))     # 0.21
print(np.sum(lasso.coef_ != 0))  # 使用された特徴量の数　4/104
# 適合不足
# alpha = 0.01, max_iter = 100000(最大の繰り返し回数)とすると0.90, 0.77
# の結果が得られるようになる。このとき、特徴量は33個使用される


# クラス分類のための線形モデル
# ロジスティック回帰と線形サポートベクターマシンは、
# 最も一般的な線形クラス分類アルゴリズムである

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(clf.__class__.__name__)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()
plt.show()
# この二つにおける正則化の強度を決定するトレードオフパラメータは
# Cと呼ばれ、これを大きくすれば正則化は弱くなる


# 線形モデルによる多クラス分類
# １対その他アプローチをつかう
# つまり、各クラスに対して、その他すべてのクラスとの分類を行う
# ガウス分布でサンプリングした二次元データセットを用いてみる
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()

linear_svc = LinearSVC().fit(X, y)
print(linear_svc.coef_.shape)       # (3, 2)
print(linear_svc.intercept_.shape)  # (3,)


# 利点、欠点、パラメータ
# 線形モデルの主要なパラメータは、回帰モデルでは、alpha、
# LinearSVCとLogisticRegressionではCと呼ばれる
# 正則化パラメータである。alphaが小さいほど、また、
# Cが大きいほど、複雑なモデル対応する。
# もう一つ決めなければならないことはL1正則化をつかうか
# L2正則化を使うかである。一部の特徴量だけが重要だと思うならば
# L1を使い、そうでなければデフォルトでL2を使うとよい。