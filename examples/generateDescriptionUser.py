import os
import sys
import asyncio
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_description.cldes import CLDES
from api.promptGemini import PromptGemini as gemini

def load_and_preprocess_data(dataset_loader):
    data = dataset_loader()
    df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                      columns=data['feature_names'] + ['cluster'])
    X = df.drop("cluster", axis=1)
    
    scaler = MinMaxScaler()
    for col in X.columns:
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
    
    return X, df

def perform_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    predicted = kmeans.predict(X)
    return kmeans, predicted

def describe_clusters(X, predicted, kmeans):
    groups = [i for i in range(len(X.columns))]
    cldes = CLDES(0.01, 10, kmeans)
    cldes.permutation_feature_importance(X, predicted, groups=groups)
    
    cluster_descriptions = []
    for cluster in np.unique(predicted):
        descriptionUser, columns_sorted = cldes.get_cluster_description(data=X, labels=predicted, cluster=cluster, output_type="description")
        cluster_descriptions.append(descriptionUser)
        
    return cluster_descriptions, columns_sorted

def format_cluster_descriptions(cluster_descriptions):
    return gemini.format_cluster_descriptions(cluster_descriptions)

async def main_workflow(dataset_loader):
    X, df = load_and_preprocess_data(dataset_loader)
    kmeans, predicted = perform_clustering(X)
    cluster_descriptions, columns_sorted = describe_clusters(X, predicted, kmeans)
    formatted_output = format_cluster_descriptions(cluster_descriptions)
    
    gemini_instance = gemini(formatted_output, columns_sorted)
    await gemini_instance.generate()

#asyncio.run(main_workflow(load_iris))
asyncio.run(main_workflow(load_wine))
