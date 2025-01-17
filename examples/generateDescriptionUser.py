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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_description.cldes import CLDES
#from api.promptGemini import PromptGemini as gemini
from cluster_description.cluster_parser import ClusterParser

def clear_terminal():
        os.system('cls' if os.name == 'nt' else 'clear')

def load_and_preprocess_data():
    automobile = fetch_ucirepo(id=320)
    
    X = pd.DataFrame(automobile.data.features, columns=automobile.data.feature_names)
    y = automobile.data.targets

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

    if not numerical_cols.empty:
        num_imputer = SimpleImputer(strategy='mean')
        X_num = pd.DataFrame(num_imputer.fit_transform(X[numerical_cols]), columns=numerical_cols)
    else:
        X_num = pd.DataFrame()

    if not categorical_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_cat = pd.DataFrame(cat_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = pd.DataFrame(encoder.fit_transform(X_cat), columns=encoder.get_feature_names_out(categorical_cols))
    else:
        X_cat_encoded = pd.DataFrame()

    if not X_num.empty and not X_cat_encoded.empty:
        X_processed = pd.concat([X_num, X_cat_encoded], axis=1)
    elif not X_num.empty:
        X_processed = X_num
    else:
        X_processed = X_cat_encoded

    return X_processed, y

def alternative1():
    X = load_and_preprocess_data()
    model = ClusterParser.menu() 
    clusterParser = ClusterParser(X, model)
    output, output_metrics = clusterParser.process_dataset()
    
    print(output)
    
def alternative2():
    X, y = load_and_preprocess_data()
    
    algorithms = [SVC(), RandomForestClassifier(random_state=42), LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(random_state=42)]
    algorithm_names = ["SVC", "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier"]
    
    all_metrics = {}

    for algorithm, name in zip(algorithms, algorithm_names):
        clusterParser = ClusterParser(X, model=algorithm)   

        output, output_metrics = clusterParser.process_dataset()
        
        all_metrics[name] = output_metrics
        
        print(f"\nMétricas para o algoritmo {name}:\n{output_metrics}")

    # ClusterParser.save_results_csv(dataset_name="student_performance", all_metrics=all_metrics, filename="metrics_output_student_performance")
    # print("Métricas de todos os algoritmos salvas no CSV com sucesso.")


async def main_workflow():
    alternative2()


asyncio.run(main_workflow())


