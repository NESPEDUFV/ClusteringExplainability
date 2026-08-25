"""
Gera as regras de cada cluster com o CLEX e plota a árvore correspondente.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.tree import plot_tree

from clint import CLEX

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

data = pd.DataFrame(X, columns=['X', 'Y'])
data["cluster"] = labels_true

clex = CLEX()

# geração da árvore
clex.fit(data.drop("cluster", axis=1), data["cluster"])

# geração das regras
labels = [0, 1, 2]  # labels que se deseja gerar regras
rules_all_groups = clex.get_rules(bin_columns=None,
                                  label=labels,
                                  min_samples=0.1,
                                  mutually_exclusives=None)

# retorna uma lista de regras para cada grupo, é possível iterar sobre elas
for idx, rules in enumerate(rules_all_groups):
    print(f"Regras do Grupo {labels[idx]}")
    for rule in rules:
        print(rule)
        print("")

fig, ax = plt.subplots()
ax = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels_true, ax=ax)
legend1 = ax.legend(loc="upper left", title="Grupos")
ax.add_artist(legend1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()

# visualizar a árvore criada com as regras completas
fig = plt.figure(figsize=(40, 40))
ax = plot_tree(clex,
               filled=False,
               feature_names=data.drop("cluster", axis=1).columns,
               fontsize=12,
               label="all",
               impurity=False)

plt.show()
