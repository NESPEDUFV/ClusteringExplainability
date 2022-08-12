# Cluster Explainability

Implementação de uma forma de obter explicações automáticas de clusters gerados por algoritmos não-supervisionados. 
Utiliza CART - Classification and regression trees - e tenta gerar regras amigáveis. 

## Exemplo de utilização básico 

Para utilizar a geração de regras, inicialmente é necessário criar a árvore de decisão para os dados.  

O método `fit` realiza a criação da árvore, recebendo como parâmetros um `dataframe` com os dados utilizados para o agrupamento e uma série com o cluster de cada instância/linha, ou seja, o número indicando a qual grupo cada instância/linha pertence. É possível também passar parâmetros adicionais para criar árvores com diferentes características.

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

# criação da árvore para geração de regras
clex.fit(iris_df.drop("target", axis=1), iris["cluster"])
```

Após criar a árvore, é possível gerar as regras. O método `get_rules` é o responsável pela geração. Ele tem diversos parâmetros para geração das regras. O parâmetro `bin_columns` é utilizado para verificação de colunas binárias ou dummy, fazendo com que regras mais legíveis sejam criadas. O parâmetro `label` é uma lista com os rótulos de quais grupos se deseja gerar as regras. Com o objetivo de filtrar regras desnecessárias, com poucas amostras, o parâmetro `min_samples`pode ser utilizado. Ele recebe uma porcentagem mínima de amostras da classe para uma regra ser considerada. Por fim, existe o parâmetro `mutually_exclusives`. Ele recebe uma lista de atributos mutuamente exclusivos e é utilizado para gerar regras mais legíveis. Pode ser inserido também um conjunto de atributos mutuamente exclusivos, com uma "lista de listas" de atributos desse tipo. Ainda é possível passar as listas de atributos mutuamente exclusivos como parâmetros personalizados separados, caso eles fiquem ao final da chamada do método. Para isso, basta adicionar parâmetros com um nome e a lista de atributos que sejam mutuamente exclusivos.    

```python

# geração de regras
clex.get_rules(bin_columns=None,
                label=[0],
                min_samples=0.1,
                mutually_exclusives=None # poderia ser: [["atributo 1", "atributo 2"], ...]. atributos que nao ocorrem ao mesmo tempo
                )


# visualizar a árvore criada com as regras completas
fig = plt.figure(figsize=(40,40))
ax = plot_tree(clex, 
                filled=True, 
                feature_names=iris["feature_names"],
                fontsize=8)

plt.show()
```
