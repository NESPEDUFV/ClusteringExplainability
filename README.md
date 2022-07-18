# Cluster Explainability

Implementação de uma forma de obter explicações automáticas de clusters gerados por algoritmos não-supervisionados. 
Utiliza CART - Classification and regression trees - e tenta gerar regras amigáveis. 

## Exemplo de utilização básico 

Abaixo é possível visualizar um exemplo simples de como utilizar: 

```python

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
clex.fit(iris_df.drop("target", axis=1), iris["cluster"])
clex.get_rules(label=[0])


# visualizar a árvore criada com as regras completas
fig = plt.figure(figsize=(40,40))
ax = plot_tree(clex, 
                filled=True, 
                feature_names=iris["feature_names"],
                fontsize=8)

plt.show()
```
