# ニューラルネットワーク（ディープラーニング）

# 多層パーセプトロン（MLP）
# 重み付き和の計算が繰り返し行われる
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise = 0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# 隠れ層のユニット数を10にしたいときは
# hidden_layer_sizes=[10]
# とすればよい
# 隠れ層のユニット数が10の層を二つにしたいときは
# hidden_layer_sizes=[10, 10]
# とすればよい
# さらに、非線形活性化関数にtanhをつかうには
# activation='tanh', hidden_layer_sizes=[10, 10]
# とする
# ニューラルネットワークには複雑さを制御する方法が、隠れ層の数、
# 隠れ層のユニット数、正則化（alpha）と、ほかにもたくさん存在する。

# より柔軟な、もしくはより大きなモデルを使いたいなら
# scikit-learnをつかうのではなく、keras,lasagne,tensor-flow
# のような、ディープラーニングのライブラリを試したほうがよい

# ニューラルネットワークのパラメータ調整の一般的なやり方は
# まず、過剰適合できるように大きいネットワークを作る
# そして、タスクがそのネットワークで訓練データを学習できることを確認する
# 最後に、ネットワークを小さくするか、alphaを増やして正則化を強化して
# 汎化性能を向上させる