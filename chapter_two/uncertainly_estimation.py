# 不確実性推定
# あるテストポイントに対して、
# クラス分類器が出力する予測クラスだけではなく
# その予測がどれくらい確かなのかを知りたいことがよくある
# sklearnには、不確実性推定に利用できる関数が二つある
# decision_functionとpredict_probaである

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
y_named = np.array(["blue", "red"])[y]
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

# decision_function
# 二クラス分類ではdecision_functionの結果の配列は(n_samples,)の形になり
# サンプルごとに一つの浮動小数点が返される
# このデータ値には、あるデータポイントが
# 「陽性であると、モデルが信じている度合いがエンコードされている
# 正であれば陽性クラス、負であれば陰性クラスを意味する
print(gbrt.decision_function(X_test).shape) # (25,)
print(gbrt.decision_function(X_test)[:6])
# [ 4.13592629 -1.7016989  -3.95106099 -3.62599351  4.28986668  3.66166106]

# predict_proba
# 二クラス分類ではdecision_functionの結果の配列は(n_samples, 2)の形になる
# 各行の第一エントリは第一クラスの予測確率で
# 第二エントリは第二クラスの予測確率である
print(gbrt.predict_proba(X_test).shape) # (25, 2)
print(gbrt.predict_proba(X_test)[:6])
# [[0.01573626 0.98426374]
#  [0.84575649 0.15424351]
#  [0.98112869 0.01887131]
#  [0.97406775 0.02593225]
#  [0.01352142 0.98647858]
#  [0.02504637 0.97495363]]

# 多クラス分類の不確実性
# 上の二つの関数は、多クラス分類の場合でも使える
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)
# 形はどちらも(n_samples, n_classes)になる
print(gbrt.decision_function(X_test)[:6, :])
# [[-1.94506358 -2.01517314  0.12763239]
#  [-1.94762574  0.0495959  -1.86974405]
#  [ 0.08367196 -2.01610636 -1.86746092]
#  [-1.94509191 -2.01517314  0.12763239]
#  [ 0.08367196 -2.01610636 -1.86746092]
#  [-1.94506358 -2.01517314  0.12763239]]
print(gbrt.predict_proba(X_test)[:6, :])
# [[0.10122985 0.09437575 0.8043944 ]
#  [0.10582515 0.77977843 0.11439643]
#  [0.79076603 0.09685585 0.11237812]
#  [0.10122728 0.09437602 0.8043967 ]
#  [0.79076603 0.09685585 0.11237812]
#  [0.10122985 0.09437575 0.8043944 ]]
