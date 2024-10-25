import os
import sys
import asyncio
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, load_digits
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_description.cldes import CLDES
#from api.promptGemini import PromptGemini as gemini
from cluster_description.cluster_parser import ClusterParser

def clear_terminal():
        os.system('cls' if os.name == 'nt' else 'clear')

def load_and_preprocess_data(dataset_loader):
    data = dataset_loader()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['cluster'] = data.target
    
    X = df.drop("cluster", axis=1)
    
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled

def alternative1(dataset_loader):
    X = load_and_preprocess_data(dataset_loader)
    model = ClusterParser.menu() 
    clusterParser = ClusterParser(X, model)
    output, output_metrics = clusterParser.process_dataset()
    
    print(output)
    
def alternative2(dataset_loader):
    X = load_and_preprocess_data(dataset_loader)
    
    algorithms = [SVC(), RandomForestClassifier(random_state=42), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=42)]
    algorithm_names = ["SVC", "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier"]
    
    all_metrics = {}

    for algorithm, name in zip(algorithms, algorithm_names):
        clusterParser = ClusterParser(X, model=algorithm)   

        output, output_metrics = clusterParser.process_dataset()
        
        all_metrics[name] = output_metrics
        
        print(f"\nMétricas para o algoritmo {name}:\n{output_metrics}")

    ClusterParser.save_results_csv(dataset_name="Digits", all_metrics=all_metrics, filename="metrics_output_digits.csv")
    print("Métricas de todos os algoritmos salvas no CSV com sucesso.")

def check_values_in_range(df, column_name, lower_bound=0, upper_bound=1):
    """
    Verifica se há algum valor na coluna especificada do DataFrame que está entre os limites especificados.

    Parameters:
    - df: pd.DataFrame - O DataFrame que contém os dados.
    - column_name: str - O nome da coluna a ser verificada.
    - lower_bound: float - O limite inferior (padrão: 0).
    - upper_bound: float - O limite superior (padrão: 1).

    Returns:
    - bool - True se houver pelo menos um valor dentro do intervalo, caso contrário, False.
    - list - Lista dos valores que estão dentro do intervalo (se desejado).
    """
    
    if column_name not in df.columns:
        raise ValueError(f"A coluna '{column_name}' não existe no DataFrame.")
    
    # Verifica os valores que estão entre lower_bound e upper_bound
    values_in_range = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    
    # Retorna um booleano e uma lista com os valores
    return not values_in_range.empty, values_in_range[column_name].tolist()

async def main_workflow(dataset_loader):
    alternative1(dataset_loader)


asyncio.run(main_workflow(load_digits))


