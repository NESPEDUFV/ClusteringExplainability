import pandas as pd
import numpy as np
from scipy.stats import entropy

class CLDES:

    def __init__(self, min_per, num_bins = -1, ) -> None:
        self._min_per = min_per
        self._num_bins = num_bins
    
    def _entropy1(self, labels, base=None, counts=None):
        
        value,counts = np.unique(labels, return_counts=True)
        return entropy(counts)

    def generate_cluster_description(self, data: pd.DataFrame, cluster: int):
        data = data.dropna(axis=1)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cols = data.select_dtypes(numerics).columns
        continuos_vars = [x for x in cols if len(data[x].unique()) > 10 and data[x].dtype and x != "cluster"]
        
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
            
            print("Entropy: ", entropy(value_counts.values, ))
            print("")
            print(col)
            print(pd.DataFrame({
                "porc.": percs,
                "valores": percs_idx
            }))
            print("")


        print(data.head())