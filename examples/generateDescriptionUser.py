"""
Pipeline completo: clustering, descrição dos clusters e cálculo das métricas.

`alternative1` roda um modelo escolhido interativamente; `alternative2` compara
todos os classificadores e salva as métricas em CSV.
"""

import logging
import os
import sys

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from pymfe.mfe import MFE
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clint.cluster_parser import ClusterParser  # noqa: E402


def load_and_preprocess_data(dataset_loader):
    data = dataset_loader()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['cluster'] = data.target

    X = df.drop("cluster", axis=1)

    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


def alternative1(dataset_loader):
    X = load_and_preprocess_data(dataset_loader)
    model = ClusterParser.menu()
    cluster_parser = ClusterParser(X, model)
    output, _ = cluster_parser.process_dataset()

    print(output)


def alternative2(dataset_loader, dataset_name="Digits", filename="metrics_output_digits.csv"):
    X = load_and_preprocess_data(dataset_loader)

    algorithms = {
        "SVC": SVC(),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    }

    all_metrics = {}

    for name, algorithm in algorithms.items():
        cluster_parser = ClusterParser(X, model=algorithm)
        _, output_metrics = cluster_parser.process_dataset()

        all_metrics[name] = output_metrics

        print(f"\nMétricas para o algoritmo {name}:\n{output_metrics}")

    ClusterParser.save_results_csv(dataset_name=dataset_name, all_metrics=all_metrics, filename=filename)
    print("Métricas de todos os algoritmos salvas no CSV com sucesso.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    alternative1(load_digits)
