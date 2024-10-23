import os
import sys
import asyncio
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_description.cldes import CLDES
#from api.promptGemini import PromptGemini as gemini
from cluster_description.cluster_parser import ClusterParser

def clear_terminal():
        os.system('cls' if os.name == 'nt' else 'clear')

def load_and_preprocess_data_iris(dataset_loader):
    data = dataset_loader()
    df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                        columns=data['feature_names'] + ['cluster'])
    X = df.drop("cluster", axis=1)
    
    scaler = MinMaxScaler()
    for col in X.columns:
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
    
    return X

async def main_workflow(dataset_loader):
    X = load_and_preprocess_data_iris(load_iris)
    model = ClusterParser.menu() 
    clusterParser = ClusterParser(X, model)
    output, output_metrics = clusterParser.process_dataset()
    
    clear_terminal()
    print(output_metrics)
    
    algorithm_name = type(model).__name__  
    dataset_name = "Iris"
    ClusterParser.save_results(algorithm_name, dataset_name, output_metrics)

asyncio.run(main_workflow(load_diabetes))

