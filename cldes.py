import pandas as pd
from pandas.api.types import is_object_dtype
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class CLDES:

    def __init__(self, min_per, num_bins = -1, ) -> None:
        self._min_per = min_per
        self._num_bins = num_bins
    
    def generate_cluster_description(self, data: pd.DataFrame, cluster: int):
        percentages = {}
        continuos_vars = [x for x in data.columns if len(data[x].unique()) > 10]
        
        for col in continuos_vars: 
            bins = np.histogram_bin_edges(data[col], bins="fd")
            data[col] = pd.cut(data[col], bins)

        data = data[data["cluster"] == cluster]
        
        for col in data.columns:
            value_counts = data[col].value_counts()
            percs = value_counts/data.shape[0]
            percs_idx = percs[percs >= self._min_per].index
            percs = percs[percs_idx].values
            values = value_counts[percs_idx].values    

            print("atributo: ", col)
            print(pd.DataFrame({
                "porc.": percs,
                "valores": percs_idx
            }))
            print("")


        print(data.head())