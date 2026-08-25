"""
Gera a descrição de um cluster do dataset Iris usando o CLDES.
"""

<<<<<<< Updated upstream:examples/cldes.py
from cluster_description.cldes import CLDES
from cluster_description.clex import CLEX
from sklearn.datasets import load_iris, load_wine
import pandas as pd
=======
>>>>>>> Stashed changes:examples/example_cldes.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from clint import CLDES, OutputType

if __name__ == "__main__":
    iris = load_iris()
    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['cluster'])

    X = iris_df.drop("cluster", axis=1)

    scaler = MinMaxScaler()
    for i in X.columns:
        X[i] = scaler.fit_transform(X[i].values.reshape(-1, 1))

    kmeans = KMeans(3).fit(X)
    predicted = kmeans.predict(X)

    groups = [i for i in range(len(X.columns))]

    cldes = CLDES(0, 0)
    cldes.permutation_feature_importance(X, predicted, groups=groups)
    description, columns_sorted = cldes.get_cluster_description(X, predicted, 0, OutputType.PREDICATES)

    print(f"Atributos por importância: {list(columns_sorted)}\n")
    for predicate in description:
        print(predicate)
