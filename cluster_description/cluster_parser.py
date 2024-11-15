import os
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import csv

from cluster_description.cldes import CLDES

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.promptGemini import PromptGemini as gemini

class ClusterParser:
    def __init__(self, df, model, n_clusters=3):
        self.n_clusters = n_clusters
        self.cldes = CLDES(0, 0.01, model)
        self.X = df
        self.df = None
        self.predicted = None
        self.cluster_descriptions = None
        self.formatted_output = None
        self.metrics = None
        
    def menu():
        while True:
            print("Escolha o algoritmo para o teste:")
            print("1. SVM")
            print("2. Random Forest")
            print("3. Regressão Logística")
            print("4. KNN")
            print("5. Árvore de Decisão")
            escolha = input("Digite o número correspondente ao algoritmo: ")
            
            if escolha == '1':
                return SVC()
            elif escolha == '2':
                return RandomForestClassifier(random_state=42)
            elif escolha == '3':
                return LogisticRegression()
            elif escolha == '4':
                return KNeighborsClassifier()
            elif escolha == '5':
                return DecisionTreeClassifier(random_state=42)
            else:
                print("Escolha inválida. Tente novamente.")

    def perform_clustering(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42).fit(self.X)
        self.predicted = kmeans.predict(self.X)
        return self.predicted

    def perform_clustering_dbscan(self, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(self.X)
        self.predicted = dbscan.labels_
        return dbscan, self.predicted

    def describe_clusters(self):
        groups = [i for i in range(len(self.X.columns))]

        self.cldes.permutation_feature_importance(self.X, self.predicted, groups=groups)
        
        self.cluster_descriptions = []
        for cluster in np.unique(self.predicted):
            descriptionUser, columns_sorted = self.cldes.get_cluster_description(data=self.X, labels=self.predicted, cluster=cluster, output_type="description")
            self.cluster_descriptions.append(descriptionUser)
            
        print(self.cluster_descriptions)
        return self.cluster_descriptions

    def format_cluster_descriptions(self):
        self.formatted_output = gemini.format_cluster_descriptions(self.cluster_descriptions)
        return

    def calculate_metrics_for_all_clusters(self):
        clusters = self.parse_formatted_output(self.formatted_output)
        metrics_output = {}

        for cluster, rule_str_list in clusters.items():
            rules = [self.parse_rule(rule_str) for rule_str in rule_str_list]
            
            coverage = self.cldes.calculate_coverage(rules, self.X, self.predicted, cluster)
            separation_error = self.cldes.calculate_separation_error(rules, self.X, self.predicted, cluster)
            conciseness = self.cldes.calculate_conciseness(rule_str_list)
            
            metrics_output[cluster] = {
                "coverage": coverage,
                "separation_error": separation_error,
                "conciseness": conciseness
            }

        self.metrics = metrics_output
        return self.metrics

    
    def format_clusters_and_metrics(self) -> None:
        self.formatted_output += "\n\n"
        
        for cluster, metrics in self.metrics.items():
            self.formatted_output += (
                f"Métricas para o Cluster {cluster}:\n"
                f"coverage: {metrics['coverage']:.4f}\n"
                f"separation error: {metrics['separation_error']:.4f}\n"
                f"conciseness: {metrics['conciseness']:.4f}\n\n"
            )
        
        self.formatted_output = self.formatted_output.strip()
        return

    def parse_rule(self, rule_str):
        """
        Função para transformar uma string de regra em um predicado.
        """
        parts = rule_str.strip('<>').split(", ", 2)

        if len(parts) == 3:
            attribute, rule_type, values_str = parts
            if rule_type == "contains":
                value = eval(values_str) 
                return lambda x: x[attribute] == value
            elif rule_type == "80-between": 
                values = eval(values_str)
                lower_bound, upper_bound = values
                return lambda x: lower_bound <= x[attribute] <= upper_bound
            else:
                raise ValueError(f"Tipo de regra não reconhecido: {rule_type}")
        else:
            raise ValueError(f"Formato inesperado na regra: {rule_str}")

    def parse_formatted_output(self, formatted_output):
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

    def save_results(algorithm_name, dataset_name, formatted_metrics, filename="results1.txt"):
        file_exists = os.path.isfile(filename)

        with open(filename, 'a') as file:
            if not file_exists:
                file.write("\n\nAlgoritmo, Dataset, Resultados\n")

            file.write(f"{algorithm_name}, {dataset_name}, {formatted_metrics}\n")
            file.write("\n")
            
    def save_results_csv(dataset_name, all_metrics, folder="comparacaoResultados/csv", filename="metrics_output.csv"):

        os.makedirs(folder, exist_ok=True)
        
        filepath = os.path.join(folder, filename)
        
        with open(filepath, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Algoritmo", "Dataset", "Cluster", "Coverage", "Separation Error", "Conciseness"])
            
            for algorithm, clusters_metrics in all_metrics.items():
                for cluster, metrics in clusters_metrics.items():
                    writer.writerow([
                        algorithm,
                        dataset_name,
                        cluster,
                        metrics['coverage'],
                        metrics['separation_error'],
                        metrics['conciseness']
                    ])

        print(f"Arquivo salvo com sucesso em: {filepath}")

    def process_dataset(self):
        self.perform_clustering()
        self.describe_clusters()
        self.format_cluster_descriptions()
        self.calculate_metrics_for_all_clusters()
        self.format_clusters_and_metrics()
        return self.formatted_output, self.metrics