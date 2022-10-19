import numpy as np
import sys

# adicionar caminho do script
path = "/home/guilherme/Documentos"
sys.path.insert(0, path + "/cluster_explainability")
from cldes import CLDES
from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['cluster'])

cldes = CLDES(0, 10)

cldes.generate_cluster_description(iris_df, 1)

#df = pd.read_csv("/home/guilherme/Documentos/porto_seguro/data/personas_new.csv", index_col=0)
#print(df.head())

#print(df.info())

#cldes.generate_cluster_description(df.drop("cod_pessoa", axis=1), 1)

