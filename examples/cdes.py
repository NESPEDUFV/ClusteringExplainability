import numpy as np
import sys

# adicionar caminho do script
path = "/home/guilherme/Documentos"
sys.path.insert(0, path + "/cluster_explainability")
from cldes import CLDES
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.tree import plot_tree


iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['cluster'])

cldes = CLDES(0.1, 10)

cldes.generate_cluster_description(iris_df, 1)

