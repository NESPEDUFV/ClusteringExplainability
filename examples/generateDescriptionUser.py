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
    
    return X, df
def load_and_preprocess_data_wine(dataset_loader):
    data = dataset_loader()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    df['target'] = data.target
    
    df.rename(columns={'od280/od315_of_diluted_wines': 'Absorbância_280_vs_315'}, inplace=True)
    
    X = df.drop("target", axis=1)
    
    scaler = MinMaxScaler()
    for col in X.columns:
        X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
    
    return X, df

def perform_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    predicted = kmeans.predict(X)
    return kmeans, predicted

def describe_clusters(X, predicted, cldes):
    groups = [i for i in range(len(X.columns))]

    cldes.permutation_feature_importance(X, predicted, groups=groups)
    
    cluster_descriptions = []
    for cluster in np.unique(predicted):
        descriptionUser, columns_sorted = cldes.get_cluster_description(data=X, labels=predicted, cluster=cluster, output_type="description")
        cluster_descriptions.append(descriptionUser)
        
    return cluster_descriptions, columns_sorted

def format_cluster_descriptions(cluster_descriptions):
    return gemini.format_cluster_descriptions(cluster_descriptions)

def calculate_metrics_for_all_clusters(cldes, X, predicted, formatted_output):
    clusters = parse_formatted_output(formatted_output)
    metrics_output = []

    formatted_output += "\n\n"
    
    for cluster, rule_str_list in clusters.items():
        rules = [parse_rule(rule_str) for rule_str in rule_str_list]
        
        coverage = cldes.calculate_coverage(rules, X, predicted, cluster)
        separation_error = cldes.calculate_separation_error(rules, X, predicted, cluster)
        conciseness = cldes.calculate_conciseness(rule_str_list)
        
        metrics_output.append(f"Métricas para o Cluster {cluster}:\n"
                              f"coverage: {coverage}\n"
                              f"separation error: {separation_error}\n"
                              f"conciseness: {conciseness}\n")
    
    formatted_output += "\n".join(metrics_output)
    return formatted_output

def parse_rule(rule_str):
    """
    Função para transformar uma string de regra em um predicado.
    """
    parts = rule_str.strip('<>').split(", ", 2)
    
    if len(parts) == 3:
        attribute, rule_type, values_str = parts
        values = eval(values_str)
        lower_bound, upper_bound = values
        
        return lambda x: lower_bound <= x[attribute] <= upper_bound
    else:
        raise ValueError(f"Formato inesperado na regra: {rule_str}")
    
def parse_formatted_output(formatted_output):
    """
    Parseia a saída formatada, organizando as regras por cluster.
    """
    clusters = {}
    current_cluster = None
    for line in formatted_output.splitlines():
        if line.startswith("cluster"):
            current_cluster = int(line.split()[1][:-1])
            clusters[current_cluster] = []
        elif line:
            clusters[current_cluster].append(line.strip('<>'))
    return clusters

async def main_workflow(dataset_loader):
    X, df = load_and_preprocess_data_wine(dataset_loader)
    kmeans, predicted = perform_clustering(X)
    
    cldes = CLDES(0.01, 10, kmeans)
    
    cluster_descriptions, columns_sorted = describe_clusters(X, predicted, cldes)
    formatted_output = format_cluster_descriptions(cluster_descriptions)
    
    formatted_output = calculate_metrics_for_all_clusters(cldes, X, predicted, formatted_output)
    
    clear_terminal()
    print(formatted_output)
    gemini_instance = gemini(formatted_output, columns_sorted)
    await gemini_instance.generate()

asyncio.run(main_workflow(load_wine))

