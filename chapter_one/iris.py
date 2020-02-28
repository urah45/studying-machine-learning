from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt


iris_dataset = load_iris()

print(iris_dataset.keys())
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(iris_dataset['target_names'])
# ['setosa' 'versicolor' 'virginica']



# データの分割
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# データをよく確認する
# 例えば、もしかしたら、cmではなくインチかもしれない
# 散布図のような可視化をすると良い

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
# plt.show()


# k-最近傍法
# k個の近傍点を用いることができることを意味している
knn = KNeighborsClassifier(n_neighbors=1) # ここではk=1とした
knn.fit(X_train, y_train)
print(round(knn.score(X_test, y_test),3))

