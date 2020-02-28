# 決定木
# 決定木はクラス分類と回帰タスクに広く用いられているモデルである


# クラス分類
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print(tree.score(X_train, y_train))  # 1.000
print(tree.score(X_test, y_test))    # 0.937
# 過剰適合

# 事前枝刈り
# 構築過程で木の生成を早めに止めてしまうことで過剰適合を防ぐ
tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_train, y_train)
print(tree.score(X_train, y_train))  # 0.988
print(tree.score(X_test, y_test))    # 0.951
# 事後枝刈りという、情報の少ないノードを削除する方法もある

# 特徴量の重要度
print(tree.feature_importances_)
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.01019737 0.04839825
#  0.         0.         0.0024156  0.         0.         0.
#  0.         0.         0.72682851 0.0458159  0.         0.
#  0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]


# 回帰
# 決定木によるモデルを回帰に使う際に注意しなければならないことがある
# それは、外挿ができない、つまり、訓練データの外側に対しては予測ができない
# 教科書P.81図2-32を見ればわかる

# 長所、短所、パラメータ
# 長所は、結果のモデルが容易に可視化可能でｍ専門家でなくても理解可能であること
# さらに、データのスケールに対して完全に不変であることである
# 特徴量の正規化や、標準化が必要ない
# 最大の問題点は、事前枝刈りを行っても過剰適合しやすく、
# 汎化性能が低い傾向があること
# このため、ほとんどの場合ではアンサンブル法が用いられる

# 決定木のアンサンブル法

# ランダムフォレスト
# 少しずつ異なる決定機を集める
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2) # 決定木5個
forest.fit(X_train, y_train)
print(forest.score(X_train, y_train))  # 0.96
print(forest.score(X_test, y_test))    # 0.92
# random forestでも重要度があり、単独の決定木よりも信用できる

# 長所、短所、パラメータ
# ランダムフォレストは完全にランダムであるので、
# 乱数シード(random_state)を変更すると構築されるモデルがおおきくかわる可能性がある
# ランダムフォレストは、テキストデータなどの非常に
# 高次元で疎なデータに対してはうまく機能しない傾向にある
# このようなデータに対しては、線形モデルのほうが適している
# さらに、ランダムフォレストは、線形モデルよりも、
# 多くのメモリを消費するし、訓練も予測も遅い
# 実行時間やメモリが重要なアプリケーションでは、線形モデルを使ったほうがよい


# 勾配ブースティング回帰木（勾配ブースティングマシン）
# 複数の決定木を組み合わせてより強力なモデルを構築するもう一つのアンサンブル学習である
# 分類も回帰にも使える
# 勾配ブースティングでは、ひとつ前の決定木の誤りを次の決定木が修正するようにして
# 決定木を順番に作っていく
# デフォルトでは、乱数性はなく、強力な事前枝刈りが行われる
# ポイントは、浅い決定木のような簡単なモデルを多数組み合わせることにある
# パラメータはlearning_rateと、n_estimators
# 一般には、初めにランダムフォレストをためし、うまくいったとしても時間がかかったり
# 最後の1%まで性能を絞り出したい場合に勾配ブースティングを試すとよい

from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print(gbrt.score(X_train, y_train))   # 1.0
print(gbrt.score(X_test, y_test))     # 0.88

# 長所、短所、パラメータ
# 勾配ブースティング回帰木は、教師あり学習の中で最も強力で
# 広く使われているモデルである
# 短所は、パラメータのチューニングに細心の注意が必要であることと
# 訓練にかかる時間が長いことである。また、高次元な疎なデータに対しては
# あまりうまく機能しない
