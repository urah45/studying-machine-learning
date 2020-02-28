import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import pandas as pd


# scipy
eye = np.eye(4)    ## 4次単位行列

# Numpy配列をScipyのCSR形式の疎行列に変換する
# 非ゼロ行列だけが格納される
sparse_matrix = sparse.csr_matrix(eye)
print(sparse_matrix)

# COO形式で疎行列を作成
data = np.ones(4) # array([1, 1, 1, 1])
row_indices = np.arange(4)  # array([0, 1, 2, 3])
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print(eye_coo)

# matplotlib
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y, marker="x")
# plt.show()    # グラフを表示

# pandas
data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Location": ["New York", "Paris", "Berlin", "London"],
    "Age": [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)
print(data_pandas)
# 30歳以上の人のデータのみを取り出す
print(data_pandas[data_pandas.Age > 30])
# Jupyter notebookで以下を実行するときれいな表になる
from IPython.display import display
display(data_pandas)

