import os
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from cluster_description.cldes import CLDES

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.promptGemini import PromptGemini as gemini

class ClusterParser:
    def __init__(self, dataset_loader, n_clusters=3):
        self.dataset_loader = dataset_loader
        self.n_clusters = n_clusters
        self.cldes = CLDES(0.01, 0)
        self.X = None
        self.df = None
        self.predicted = None
        self.cluster_descriptions = None
        self.formatted_output = None

    def clear_terminal(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def load_and_preprocess_data_iris(self):
        data = self.dataset_loader()
        df = pd.DataFrame(data=np.c_[data['data'], data['target']],
                          columns=data['feature_names'] + ['cluster'])
        X = df.drop("cluster", axis=1)
        
        scaler = MinMaxScaler()
        for col in X.columns:
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
        
        self.X = X
        self.df = df
        return X, df
    
    def load_and_preprocess_data_wine(self):
        data = self.dataset_loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        
        df['target'] = data.target
        
        df.rename(columns={'od280/od315_of_diluted_wines': 'Absorbância_280_vs_315'}, inplace=True)
        
        X = df.drop("target", axis=1)
        
        scaler = MinMaxScaler()
        for col in X.columns:
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
        
        self.X = X
        self.df = df
        return X, df

    def load_and_preprocess_data_diabetes(self):
        data = self.dataset_loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        
        df['target'] = data.target
        
        X = df.drop("target", axis=1)
        
        scaler = MinMaxScaler()
        for col in X.columns:
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
        
        self.X = X
        self.df = df
        return X, df

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
            
        return self.cluster_descriptions

    def format_cluster_descriptions(self):
        return gemini.format_cluster_descriptions(self.cluster_descriptions)

    def calculate_metrics_for_all_clusters(self):
        clusters = self.parse_formatted_output(self.formatted_output)
        metrics_output = []

        self.formatted_output += "\n\n"
        
        for cluster, rule_str_list in clusters.items():
            rules = [self.parse_rule(rule_str) for rule_str in rule_str_list]
            
            coverage = self.cldes.calculate_coverage(rules, self.X, self.predicted, cluster)
            separation_error = self.cldes.calculate_separation_error(rules, self.X, self.predicted, cluster)
            conciseness = self.cldes.calculate_conciseness(rule_str_list)
            
            metrics_output.append(f"Métricas para o Cluster {cluster}:\n"
                                  f"coverage: {coverage}\n"
                                  f"separation error: {separation_error}\n"
                                  f"conciseness: {conciseness}\n")
        
        self.formatted_output += "\n".join(metrics_output)
        return self.formatted_output

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
            else:
                values = eval(values_str)
                lower_bound, upper_bound = values
                return lambda x: lower_bound <= x[attribute] <= upper_bound
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

    def load_and_preprocess_data_breast_cancer(self):
        data = self.dataset_loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        
        df['target'] = data.target
        
        X = df.drop("target", axis=1)
        
        scaler = MinMaxScaler()
        for col in X.columns:
            X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
        
        self.X = X
        self.df = df
        return X, df

    def process_dataset(self):
        self.load_and_preprocess_data_diabetes()
        self.perform_clustering()
        self.describe_clusters()
        self.formatted_output = self.format_cluster_descriptions()
        self.formatted_output = self.calculate_metrics_for_all_clusters()
        return self.formatted_output
