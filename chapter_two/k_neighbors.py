# k-最近傍法
# 訓練データセットの中から一番近い点を見つける

import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# mglearn.plots.plot_knn_classification(n_neighbors=1)
# plt.show()

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

fig, axes = plt.subplots(1, 4, figsize=(10, 3))  # 行, 列, サイズ

for n_neighbors, ax in zip([1,3,9,20], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True,eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title(str(n_neighbors) + "neighbor(s)")
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()



# ===========================================================



# k-近傍回帰

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
import numpy as np


X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))
# R^2スコアを返す
#R^2スコアとは、決定係数とも呼ばれ、回帰モデルの予測の正確さを測る指標で、
# 0から1の値をとる

fig, axes = plt.subplots(1,3, figsize=(15,4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title(str(n_neighbors) + 'neighbor(s)\n' 
                    + 'train score:' + str(round(reg.score(X_train, y_train),3))
                    + 'test score:' + str(round(reg.score(X_test, y_test),3))
                )
    ax.set_xlabel('feature')
    ax.set_ylabel('target')
    
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"])
plt.show()



# ===================================================================



# k-最近傍法アルゴリズムは、理解がしやすいが処理速度が遅く、
# 多数の特徴量を扱うことができないため、実際にはほとんど使われていない