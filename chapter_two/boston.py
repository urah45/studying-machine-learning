# 実世界の回帰データセットとして
# boston_housingデータセット
# をもちいる

from sklearn.datasets import load_boston
import mglearn

boston = load_boston()
print(boston.data.shape)
# (506, 13)

# このデータには13の特徴量があるが、ここでは13の測定結果だけではなく
# 特徴量間の積（交互作用）もみることにする。このように導出された特徴量を含めることを
# 特徴量エンジニアリングと呼ぶ
# 以下のようにして実行できる
X, y = mglearn.datasets.load_extended_boston()
print(X.shape)
# (506, 104)
# 13 + (13C2 + 13) = 13 + 91 = 104

 

