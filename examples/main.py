
import matplotlib.pyplot as plt
import numpy as np
import sys

# adicionar caminho do modulo
path = "/home/guilherme/Documentos"
sys.path.insert(0, path + "/cluster_explainability")
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
labels = [1, 2] # labels que se deseja gerar regras
rules = clex.get_rules(label=labels)

for idx, i in enumerate(rules): 
    print(f"Amostras Classe {labels[idx]}")
    print("")
    for j in i:
        print(j)
        print("")

    print("")
# visualizar a árvore criada com as regras completas
fig = plt.figure(figsize=(40,40))
ax = plot_tree(clex, 
                filled=True, 
                feature_names=iris["feature_names"],
                fontsize=8)

plt.show()
