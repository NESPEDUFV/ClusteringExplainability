import os
import sys
import asyncio
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, load_digits
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from pymfe.mfe import MFE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_description.cldes import CLDES
#from api.promptGemini import PromptGemini as gemini
from cluster_description.cluster_parser import ClusterParser

def clear_terminal():
        os.system('cls' if os.name == 'nt' else 'clear')

def load_and_preprocess_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=["target"])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y

def analyze_class_complexity(X, y, dataset_name):
    print(f"\n--- Analisando o Dataset: {dataset_name} ---")
    
    mfe = MFE(groups=["complexity"], summary="mean")
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()
    
    mfe.fit(X_np, y_np)
    
    ft_names, ft_values = mfe.extract()
    
    temp_results = pd.Series(data=ft_values, index=ft_names)
    
    metrics_to_show = ["l1.mean", "l2.mean", "l3.mean"]

    return temp_results[metrics_to_show]
    

def alternative1():
    X, y = load_and_preprocess_data()

    metricas = analyze_class_complexity(X, y, "Iris")

    model = SVC(kernel='linear', random_state=42)
    clusterParser = ClusterParser(X, model)
    descricoes, metricas_cluster_box = clusterParser.process_dataset()
    
    clear_terminal()
    print("\nMétricas do Dataset:")
    print(metricas)
    print("\nDescrições dos Clusters:")
    print(descricoes)
    print("\nMétricas dos Clusters:")
    for cluster_id, metrics in metricas_cluster_box.items():
        print(f"Cluster {cluster_id}: {metrics}")

def alternative2():
    X, y = load_and_preprocess_data()
    
    algorithms = [SVC(kernel='linear', random_state=42), RandomForestClassifier(random_state=42), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=42)]
    algorithm_names = ["SVC", "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier"]

    all_metrics = {}

    metricas = analyze_class_complexity(X, y, "Iris")

    for algorithm, name in zip(algorithms, algorithm_names):
        clusterParser = ClusterParser(X, model=algorithm)   
        descricoes, metricas_cluster_box = clusterParser.process_dataset()
        all_metrics[name] = {
            "cluster_descriptions": descricoes,
            "cluster_metrics": metricas_cluster_box
        }

    clear_terminal()
    print("\nMétricas do Dataset:")
    print(metricas)
    print("\nMétricas e Descrições dos Clusters para todos os algoritmos:")
    for algo_name, results in all_metrics.items():
        print(f"\n--- Algoritmo: {algo_name} ---")
        print("Descrições dos Clusters:")
        for idx, desc in enumerate(results["cluster_descriptions"]):
            print(f"Cluster {idx}: {desc}")
        print()
        print("Métricas dos Clusters:")
        for cluster_id, metrics in results["cluster_metrics"].items():
            print(f"Cluster {cluster_id}: {metrics}")

    # ClusterParser.save_results_csv(dataset_name="student_performance", all_metrics=all_metrics, filename="metrics_output_student_performance")
    # print("Métricas de todos os algoritmos salvas no CSV com sucesso.")


async def main_workflow():
    alternative1()


asyncio.run(main_workflow())


