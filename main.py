
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from clex import CLEX
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.tree import plot_tree

# dataset de exemplo
iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['cluster'])

clex = CLEX()

# geração das regras para os clusters
clex.fit(iris_df.drop("cluster", axis=1), iris_df["cluster"])
print(clex.get_rules(label=[1]))


# visualizar a árvore criada com as regras completas
fig = plt.figure(figsize=(40,40))
ax = plot_tree(clex, 
                filled=True, 
                feature_names=iris["feature_names"],
                fontsize=8)

plt.show()
