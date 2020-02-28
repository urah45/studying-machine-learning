# 実世界の2クラス分類のデータセットとして
# ウィスコンシン乳がんデータセット
# を用いる

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

for n_neighbors in range(1, 11):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(range(1,11), training_accuracy, label="training accuracy")
plt.plot(range(1,11), test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
# plt.show()
# 最高値
clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))  # 0.946
print(clf.score(X_test, y_test))    # 0.937

# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)
print(logreg.score(X_train, y_train))  # 0.972
print(logreg.score(X_test, y_test))    # 0.937

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
print(forest.score(X_train, y_train))  # 1.000
print(forest.score(X_test, y_test))    # 0.958

# 勾配ブースティング
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_train, y_train))   # 1.000
print(gbrt.score(X_test, y_test))     # 0.958
# 過剰適合

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_train, y_train))   # 0.993
print(gbrt.score(X_test, y_test))     # 0.937

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_train, y_train))   # 0.993
print(gbrt.score(X_test, y_test))     # 0.937  なんか上と同じになった

# svm
from sklearn.svm import SVC
svc = SVC().fit(X_train, y_train)
print(svc.score(X_train, y_train))  # 1.0
print(svc.score(X_test, y_test))    # 0.629
# 過剰適合
# スケール変換
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
# ここで、上ではあえて訓練セットの最小値と同じレンジを用いる（詳しくは以後）
svc_scaled = SVC()
svc_scaled.fit(X_train_scaled, y_train)
print(svc_scaled.score(X_train_scaled, y_train))  # 0.955
print(svc_scaled.score(X_test_scaled, y_test))    # 0.937
# Cやgammaを増やして、より複雑にしていく
svc = SVC(C=1000).fit(X_train_scaled, y_train)
print(svc.score(X_train_scaled, y_train))  # 0.993
print(svc.score(X_test_scaled, y_test))    # 0.965

# ニューラルネットワーク
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print(mlp.score(X_train, y_train))  # 0.939
print(mlp.score(X_test, y_test))    # 0.916
# 精度はいいが、他のには劣る
# これは、すべての特徴量が同じ範囲におさまっていることを仮定していることを
# 満たしていないからである。理想的には、平均が0で、分散が1であることが望ましい
mean_on_train = X_train.mean(axis=0) # 平均
std_on_train = X_train.std(axis=0)   # 分散
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp_scaled = MLPClassifier(random_state=0)
mlp_scaled.fit(X_train_scaled, y_train)
print(mlp_scaled.score(X_train_scaled, y_train))  # 0.995
print(mlp_scaled.score(X_test_scaled, y_test))    # 0.951

