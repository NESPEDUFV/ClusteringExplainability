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

Após criar a árvore, é possível gerar as regras. O método `get_rules` é o responsável pela geração. Ele tem diversos parâmetros para geração das regras e retorna uma lista com as regras para cada grupo desejado. O parâmetro `bin_columns` é utilizado para verificação de colunas binárias ou dummy, fazendo com que regras mais legíveis sejam criadas. O parâmetro `label` é uma lista com os rótulos de quais grupos se deseja gerar as regras. Com o objetivo de filtrar regras desnecessárias, com poucas amostras, o parâmetro `min_samples`pode ser utilizado. Ele recebe uma porcentagem mínima de amostras da classe para uma regra ser considerada. Por fim, existe o parâmetro `mutually_exclusives`. Ele recebe uma lista de atributos mutuamente exclusivos e é utilizado para gerar regras mais legíveis. Pode ser inserido também um conjunto de atributos mutuamente exclusivos, com uma "lista de listas" de atributos desse tipo. Ainda é possível passar as listas de atributos mutuamente exclusivos como parâmetros personalizados separados, caso eles fiquem ao final da chamada do método. Para isso, basta adicionar parâmetros com um nome e a lista de atributos que sejam mutuamente exclusivos.    

```python

# geração das regras
labels = [1, 2] # grupos para os quais se deseja gerar regras
rules_all_groups = clex.get_rules(bin_columns=None,
                        label=labels,
                        min_samples=0.1,
                        mutually_exclusives=None 
                        )

```
O formato de retorno das regras é o seguinte: 

```python
>>> print(rules_all_groups)
```

    [
        array(['petal length (cm) maior que 2.45 &&\npetal width (cm) menor ou igual a 1.75 &&\npetal length (cm) menor ou igual a 4.95 &&\npetal width (cm) menor ou igual a 1.65 \nQuantidade: 47 - 94.0%'], dtype='<U192'),
        array(['petal length (cm) maior que 2.45 &&\npetal width (cm) maior que 1.75 &&\npetal length (cm) maior que 4.85 \nQuantidade: 43 - 86.0%'], dtype='<U185')
    ]


Finalizando, é possível imprimir as regras:

```python

# retorna uma lista de regras para cada grupo, é possível iterar sobre elas, como abaixo
for idx, rules in enumerate(rules_all_groups): 
    
    print(f"Regras do Grupo {labels[idx]}")
    for rule in rules:
        print(rule)
        print("")

```

    Regras do Grupo 1
    petal length (cm) maior que 2.45 &&
    petal width (cm) menor ou igual a 1.75 &&
    petal length (cm) menor ou igual a 4.95 &&
    petal width (cm) menor ou igual a 1.65 
    Quantidade: 47 - 94.0%

    Regras do Grupo 2
    petal length (cm) maior que 2.45 &&
    petal width (cm) maior que 1.75 &&
    petal length (cm) maior que 4.85 
    Quantidade: 43 - 86.0%


E também é possível visualizar a árvore criada para a geração das regras de maneira completa:

```python
# visualizar a árvore criada com as regras completas
fig = plt.figure(figsize=(40,40))
ax = plot_tree(clex, 
                filled=True, 
                feature_names=iris["feature_names"],
                fontsize=8)

plt.show()
```

--------------------------
## Cluster Description

Para utilizar a descrição dos clusters, são necessárias duas etapas: 

- Gerar a importância dos atributos
- Utilizar as importâncias dos atributos para gerar a descrição dos clusters

As duas etapas podem ser feitas da seguinte maneira: 

Inicialmente é instanciado um objeto CDES, responsável por toda a descrição. 

```python
# parametros: minimo de importancia, numero de bins (inutil) e modelo para o G2PC
cdes = CDES(0, 10, model)
```

O modelo contém alguns parâmetros, sendo o mais importante o min_per, referente a porcentagem mínima para um atributo ser considerado importante. Os outros atributos são para encontrar a importância dos atributos com as baselines.

Após isso, é preciso encontrar as importâncias dos atributos:

```python
# parametros: minimo de importancia, numero de bins (inutil) e modelo para o G2PC
pct_chg, pct_chg_acc = cldes.explain_it(X_train, y_train, X_test, y_test, 1, groups)
```

Para isso, é preciso separar os dados em treino e teste, onde o label do cluster será a variável independente para o modelos supervisionado utilizado na *permutation feature importance.* Além disso, é preciso passar um parâmetro com o número de repetiçõies das permutações e uma lista com o label dos grupos.

Após isso, basta gerar a descrição dos clusters: 

```python
# parametros: dados com uma coluna "cluster" referente aos grupos, o grupo que se deseja gerar a descrição e a importância dos atributos
cldes.generate_cluster_description(data, i, pct_chg)
``

