import enum
import logging
import re
from typing import Union
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
import shap

import os
import sys


path = '/cluster_description/'

sys.path.append(path)

from .description import Description
from .predicates import Predicates

class OutputType(enum.Enum):
    PREDICATES  = "predicates. "
    DESCRIPTION = "description"

class CLDES:

    def __init__(self, min_per = 0, min_importance = 0.01, model = SVC(kernel='rbf', random_state=42)) -> None:
        self.min_importance = min_importance
        self._min_per = min_per
        self.model = model
        self.pct_chg_recall = None
        self.pct_chg_acc = None
        self.shap_importance_per_cluster = None
        self.global_shap_importance = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)    

    def get_all_clusters_description(self,
                                     data,
                                     labels,
                                     output_type: OutputType = OutputType.DESCRIPTION,
                                     importance_metric="recall"):
        """
        Generate descriptions for all clusters in the given data.

        Parameters:
        - data: The dataset containing features.
        - labels: The cluster labels for the data points.
        - output_type: The type of output to generate. Default is OutputType.DESCRIPTION.
        - importance_metric: The metric to use for determining feature importance. Default is "recall".

        Returns:
        - None
        """
        if output_type == OutputType.DESCRIPTION:
            self.logger.info("Generating %s", OutputType.DESCRIPTION)
            for cluster in np.unique(labels):
                self._generate_cluster_description(data, labels, cluster, importance_metric, output_type=OutputType.DESCRIPTION)

    def get_cluster_description(self,
                                data,
                                labels,
                                cluster,
                                output_type: OutputType = OutputType.DESCRIPTION,
                                importance_metric="recall"):
        """
        Generate a description for a specific cluster in the given data.

        Parameters:
        - data: The dataset containing features.
        - labels: The cluster labels for the data points.
        - cluster: The specific cluster for which to generate a description.
        - output_type: The type of output to generate. Default is OutputType.DESCRIPTION.
        - importance_metric: The metric to use for determining feature importance. Default is "recall".

        Returns:
        - None
        """
        self.logger.info("Generating %s", output_type)
        descriptionUser, columns_sorted = self._generate_cluster_description(data, labels, cluster, importance_metric, output_type)
        return descriptionUser, columns_sorted

    def shap_feature_importance(self, X, Y):
        """
        Calcula a importância das features usando SHAP.

        Este método substitui a permutação pela atribuição de features do SHAP,
        que é mais robusto, especialmente para modelos de alta performance.

        Parâmetros:
        - X: Matriz de features.
        - Y: Vetor alvo.

        Retorna:
        - shap_importance_per_cluster: Matriz (feature x cluster) com a importância média SHAP.
        - global_shap_importance: Array (feature) com a importância média SHAP global.
        - classification_report: Relatório de classificação do modelo.
        """
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        y_test = np.array(y_test)
        clusters = np.unique(y_train)
        
        # Garante que os dados sejam arrays NumPy para o SHAP
        x_train_np = np.array(x_train)
        x_test_np = np.array(x_test)

        # 1. Treinar o modelo (como antes)
        self.model.fit(x_train_np, y_train)

        # 2. Escolher o explainer SHAP mais adequado
        print("Criando o explainer SHAP...")
        # Se o modelo for baseado em árvore (ex: RandomForest, DecisionTree), use o TreeExplainer (muito rápido)
        if isinstance(self.model, (DecisionTreeClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(self.model)
            print("Usando TreeExplainer (rápido).")
        # Para outros modelos (SVM, LogisticRegression), use o KernelExplainer (mais lento)
        else:
            # O KernelExplainer precisa de uma função de predição e um "background" de dados.
            # Usar um resumo dos dados de treino é uma prática comum.
            background_data = shap.kmeans(x_train_np, 10) # Sumariza os dados de treino em 10 centróides
            explainer = shap.KernelExplainer(self.model.predict, background_data)
            print("Usando KernelExplainer (pode ser lento para datasets grandes).")

        # 3. Calcular os valores SHAP para os dados de teste
        print("Calculando os valores SHAP...")
        shap_values = explainer.shap_values(x_test_np)
        
        # Para problemas binários, shap_values pode não ser uma lista. Padronizamos para lista.
        if not isinstance(shap_values, list):
            # Para o caso binário, explainer.shap_values(X) retorna um único array. 
            # A convenção é que os valores se referem à classe positiva (classe 1).
            # Criamos uma lista com os valores para a classe 0 (negativos) e classe 1 (positivos).
            shap_values = [-shap_values, shap_values]


        # 4. Processar os valores SHAP para preencher as matrizes de saída
        num_features = x_train.shape[1]
        num_clusters = len(clusters)

        shap_importance_per_cluster = np.zeros((num_features, num_clusters))
        global_shap_importance = np.zeros(num_features)

        # for cluster_idx, cluster_label in enumerate(clusters):
        #     # shap_values é uma lista onde o índice corresponde ao rótulo da classe
        #     # Pegamos o valor absoluto, pois queremos a magnitude do impacto
        #     mean_abs_shap_for_cluster = np.abs(shap_values[cluster_label]).mean(axis=0)
        #     shap_importance_per_cluster[:, cluster_idx] = mean_abs_shap_for_cluster

        for cluster_idx, cluster_label in enumerate(clusters):
            shap_vals = shap_values[cluster_label]

            # Caso o SHAP retorne 3D (ex: (n_samples, n_features, n_classes))
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[..., cluster_idx]  # seleciona a classe correta

            # Calcula a importância média absoluta por feature
            mean_abs_shap_for_cluster = np.abs(shap_vals).mean(axis=0)

            # Se ainda houver mais de uma dimensão (ex: (n_features, 2)), faz a média sobre as classes
            if mean_abs_shap_for_cluster.ndim > 1:
                mean_abs_shap_for_cluster = mean_abs_shap_for_cluster.mean(axis=-1)

            shap_importance_per_cluster[:, cluster_idx] = mean_abs_shap_for_cluster

        # A "importância global" (equivalente ao seu pct_chg_acc) pode ser a média da importância entre todos os clusters
        global_shap_importance = np.mean(shap_importance_per_cluster, axis=1)

        # Manter a compatibilidade da saída
        y_predicted = self.model.predict(x_test_np)
        report = classification_report(y_test, y_predicted)
        
        # As saídas agora são as matrizes de importância SHAP
        self.shap_importance_per_cluster = shap_importance_per_cluster
        self.global_shap_importance = global_shap_importance

        return shap_importance_per_cluster, global_shap_importance, report

    def permutation_feature_importance(self, X, Y, groups, n_repeats=5):
        """
        Calculate permutation feature importance for the given data.

        Parameters:
        - X: Feature matrix.
        - Y: Target vector.
        - groups: Feature groups for permutation.
        - n_repeats: Number of permutations. Default is 5.

        Returns:
        - pct_chg_recall: Change in recall for each group and cluster.
        - pct_chg_acc: Change in accuracy for each group.
        """
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        y_test = np.array(y_test)
        clusters = np.unique(y_train)
        self.model.fit(np.array(x_train), y_train)

        # print("Dados do model -- acurácia, recall, precisão, f1-score")
        # print(self.model.score(np.array(x_test), y_test))


        # preallocate output matrix number of repeats x number of features
        pct_chg = np.zeros((n_repeats, len(np.unique(groups)), len(clusters)))
        pct_chg_acc = np.zeros((n_repeats, len(np.unique(groups))))
        pct_chg_recall = np.zeros((len(np.unique(groups)), len(clusters)))

        y_predicted = self.model.predict(x_test)
        e_original = accuracy_score(y_test, y_predicted)
        r_original = recall_score(y_test, y_predicted, average=None)
        print("Original accuracy: ", e_original)
        print("Original recall: ", r_original)
        self.logger.info("Initial model accuracy: %s", e_original)
        self.logger.info("Starting permutation")
        for j in np.unique(groups):
            for k in range(n_repeats):
                x_test_copy = np.copy(x_test)
                x_test_copy[:] = x_test[:]

                # permute features
                features_to_permute = np.squeeze(list(groups == j * np.ones_like(groups)))
                sub_data = np.random.permutation(x_test_copy[:, features_to_permute])

                # include new permuted features
                x_test_copy[:, features_to_permute] = sub_data

                y_copy = self.model.predict(x_test_copy)
                e_new = accuracy_score(y_test, y_copy)
                r_new = recall_score(y_test, y_copy, average=None)
                pct_chg_acc[k, j] = e_original - e_new
                for g in clusters.astype(int):
                    pct_chg_recall[j, int(g)] += r_new[g]
                    cluster = y_test == g
                    pct_chg[k, j, int(g)] = np.sum(y_copy[cluster] != y_test[cluster]) / len(y_test[cluster])

        pct_chg_recall = pct_chg_recall / n_repeats
        pct_chg_recall = r_original - pct_chg_recall
        self.pct_chg_recall = pct_chg_recall
        self.pct_chg_acc = pct_chg_acc
        return pct_chg_recall, pct_chg_acc, classification_report(y_test, y_predicted)

    def group_permutation_change(self, X, Y, n_repeats, groups, cluster, check_var):
        """
        Calculate the percent change for each group through permutation.

        Parameters:
        - X: Feature matrix.
        - Y: Target vector.
        - n_repeats: Number of permutations.
        - groups: Feature groups for permutation.
        - cluster: The specific cluster for which to calculate the change.
        - check_var: Flag to check for variance in permutations.

        Returns:
        - Pct_Chg: Percent change matrix.
        - (optional) R: Ratio of cluster label changes.
        - (optional) VarData: Variance of permuted values for each sample and group.
        - (optional) MeanData: Mean of permuted values for each sample and group.
        """
        Pct_Chg = np.zeros((n_repeats, len(np.unique(groups))))  # preallocate output matrix number of repeats x number of features

        if check_var == 1:
            R = np.zeros((np.shape(X)[0], len(np.unique(groups))))  # preallocate matrix of ratio of cluster label changes to number of repeats
            VarData = np.zeros_like(R)  # preallocate matrix of variance of permuted values for each sample and group
            MeanData = np.zeros_like(R)

        for j in np.unique(groups):  # for each feature group
            if check_var == 1:
                Record = np.zeros((np.shape(X)[0], np.sum(list(groups == j * np.ones_like(groups))), n_repeats))  # N Samples x N Features per Group x N Repeats
            for k in range(n_repeats):  # for each repeat
                np.random.seed(seed=k)
                X_2 = np.copy(X)
                X_2[:] = X[:]  # duplicate data array

                Sub_Data = np.random.permutation(X_2[:, np.squeeze(list(groups == j * np.ones_like(groups)))])  # shuffle feature
                X_2[:, np.squeeze(list(groups == j * np.ones_like(groups)))] = Sub_Data  # add shuffled data to data matrix
                if check_var == 1:
                    Record[:, :, k] = Sub_Data  # keep track of values for each permuted feature

                if j == 1 and k == 0:
                    print(X_2)
                    print(X)

                Y_CLUSTER = Y == cluster
                Y_2 = self.model.predict(X_2)

                Y_ = Y[Y_CLUSTER]  # count the percent change only for the specific group
                Y_2_ = Y_2[Y_CLUSTER]

                # calculate percent change
                Pct_Chg[k, j] = np.sum(np.array(Y_) != np.array(Y_2_)) / len(np.squeeze(Y))
                if check_var == 1:
                    R[:, j] += np.squeeze(np.array(Y) != np.array(Y_2)) / n_repeats
                    VarData[:, j] = np.mean(np.var(Record, axis=2), axis=1)
                    MeanData[:, j] = np.mean(np.mean(Record, axis=2), axis=1)

        if check_var == 1:
            return (Pct_Chg, R, VarData, MeanData)
        else:
            return Pct_Chg

    def _generate_cluster_description(self,
                                      data: pd.DataFrame,
                                      labels: Union[np.array,
                                      pd.Series,
                                      list],
                                      cluster: int,
                                      importance_type: str = "recall",
                                      output_type: str = OutputType.DESCRIPTION):
        """
        Generate a description for a specific cluster in the given data.

        Parameters:
        - data: The dataset containing features.
        - labels: The cluster labels for the data points.
        - cluster: The specific cluster for which to generate a description.
        - importance_type: The metric to use for determining feature importance. Default is "recall".
        - output_type: The type of output to generate. Default is OutputType.DESCRIPTION.

        Returns:
        - None
        """
        data = data.dropna(axis=1)
        importances = self.pct_chg_recall if importance_type == "recall" else self.pct_chg_acc
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cols = data.select_dtypes(numerics).columns
        continuos_vars = [x for x in cols if len(data[x].unique()) > 15]

        feature_importances = list(importances[:, cluster])
        self.logger.info("Feature importance for cluster %s: %s", cluster, feature_importances)
        index_sort = np.argsort(feature_importances)
        index_sort = index_sort[::-1]
        sorted_feat_im = sorted(feature_importances, reverse=True)

        cluster_rows = np.array(labels) == cluster
        data = data[cluster_rows]

        columns_sorted = data.columns[index_sort]
        self.logger.info("Features sorted by importance: %s", columns_sorted)

        data_to_df = {}
        
        generalDescription = []
        for i, col in enumerate(columns_sorted):            
            if sorted_feat_im[i] == 0 or sorted_feat_im[i] < self.min_importance:
                continue

            if col not in continuos_vars:
                value_counts = data[col].value_counts()
                percs = value_counts/data.shape[0]
                percs_idx = percs[percs >= self._min_per].index
                percs = percs[percs_idx].values
                values = value_counts[percs_idx].values

                for value, percentage in zip(percs, values):
                    description = Description.discrete_vars(col, percentage, value) \
                        if output_type == OutputType.DESCRIPTION else Predicates.contains(col, value)
                    generalDescription.append(description)
                    # print(description)
                data_to_df[col] = value_counts.index[0]
            else:
                lower = float(round(np.percentile(data[col], 1), 3))
                upper = float(round(np.percentile(data[col], 99), 3))
                
                values = f"[{lower}, {upper}]"
                data_to_df[col] = values
                description = Description.continuos_vars(col, lower, upper) \
                    if output_type == OutputType.DESCRIPTION else Predicates.percentile(col, 80, lower, upper)

                generalDescription.append(description)
        return generalDescription, columns_sorted  

    def parse_rule(self, rule):
        """
        Interpreta regras no formato: "<feature, condition, values>"
        """
        match_between = re.match(r"<([^,]+), ([^,]+), \[([^]]+)\]>", rule)  # Para regras com "80-between"
        match_contains = re.match(r"<([^,]+), ([^,]+), ([^>]+)>", rule)    # Para regras com "contains"

        if match_between:
            feature = match_between.group(1)
            condition = match_between.group(2)
            values = [float(v.strip()) for v in match_between.group(3).split(",")]
            return feature, condition, values

        elif match_contains:
            feature = match_contains.group(1)
            condition = match_contains.group(2)
            value = float(match_contains.group(3).strip()) 
            return feature, condition, [value]

        else:
            raise ValueError(f"Invalid rule format: {rule}")
        
    def get_cluster_description_shap(self,
                                    data,
                                    labels,
                                    cluster,
                                    output_type: OutputType = OutputType.DESCRIPTION):
        """
        Gera uma descrição para um cluster específico usando a importância de features
        calculada pelo SHAP.

        Parâmetros:
        - data: O dataset contendo as features.
        - labels: Os rótulos dos clusters para os pontos de dados.
        - cluster: O cluster específico para o qual gerar a descrição.
        - output_type: O tipo de saída a ser gerado. Padrão é OutputType.DESCRIPTION.

        Retorna:
        - Uma tupla contendo a descrição do cluster e as colunas ordenadas por importância.
        """
        self.logger.info("Gerando %s com importância SHAP", output_type)
        descriptionUser, columns_sorted = self._generate_cluster_description_shap(data, labels, cluster, output_type)
        return descriptionUser, columns_sorted
    
    def _generate_cluster_description_shap(self,
                                        data: pd.DataFrame,
                                        labels: Union[np.array, pd.Series, list],
                                        cluster: int,
                                        output_type: str = OutputType.DESCRIPTION):
        """
        Lógica interna para gerar a descrição do cluster com base nos valores de SHAP.
        """
        data = data.dropna(axis=1)

        # --- PRINCIPAL MUDANÇA AQUI ---
        # Em vez de escolher entre 'recall' e 'acc', usamos diretamente a matriz de importância do SHAP.
        # A matriz do SHAP tem a forma (n_features, n_clusters), então a lógica de indexação é a mesma.
        importances = self.shap_importance_per_cluster
        
        # O restante da função é praticamente idêntico
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cols = data.select_dtypes(numerics).columns
        continuos_vars = [x for x in cols if len(data[x].unique()) > 15]

        # Seleciona a coluna de importâncias para o cluster específico
        feature_importances = list(importances[:, cluster])
        self.logger.info("Importância das features (SHAP) para o cluster %s: %s", cluster, feature_importances)
        
        index_sort = np.argsort(feature_importances)[::-1]
        sorted_feat_im = sorted(feature_importances, reverse=True)

        cluster_rows = np.array(labels) == cluster
        data = data[cluster_rows]

        columns_sorted = data.columns[index_sort]
        self.logger.info("Features ordenadas por importância (SHAP): %s", columns_sorted)

        data_to_df = {}
        generalDescription = []
        
        for i, col in enumerate(columns_sorted):
            # A lógica de filtragem e descrição permanece a mesma
            if sorted_feat_im[i] <= 0 or sorted_feat_im[i] < self.min_importance:
                continue

            if col not in continuos_vars:
                value_counts = data[col].value_counts()
                percs = value_counts / data.shape[0]
                percs_idx = percs[percs >= self._min_per].index
                
                # Correção: percs.index deve ser usado para filtrar os valores
                values_filtered = value_counts.loc[percs_idx].values
                percs_filtered = percs.loc[percs_idx].values
                
                for value_name, percentage in zip(percs_idx, percs_filtered):
                    description = Description.discrete_vars(col, percentage, value_name) \
                        if output_type == OutputType.DESCRIPTION else Predicates.contains(col, value_name)
                    generalDescription.append(description)
                
                if not percs_idx.empty:
                    data_to_df[col] = percs_idx[0]
            else:
                lower = float(round(np.percentile(data[col], 1), 3))
                upper = float(round(np.percentile(data[col], 99), 3))
                
                values = f"[{lower}, {upper}]"
                data_to_df[col] = values
                description = Description.continuos_vars(col, lower, upper) \
                    if output_type == OutputType.DESCRIPTION else Predicates.percentile(col, 80, lower, upper)

                generalDescription.append(description)
                
        return generalDescription, columns_sorted
    
    def calculate_coverage(self, rules, data, labels, cluster):
        """
        Calcula o coverage de um cluster específico.
        
        Parameters:
        - rules: Lista de regras, onde cada posição representa as regras de um cluster.
        - data: DataFrame com os dados.
        - labels: Array com os rótulos dos clusters.
        - cluster: O índice do cluster para o qual calcular o coverage.
        
        Returns:
        - coverage: Proporção de amostras no cluster cobertas pelas regras.
        """

        cluster_rules = rules[cluster]
        cluster_data = data[labels == cluster]
        
        if cluster_data.empty:
            return 0.0
        
        covered_samples = set()
        for rule in cluster_rules:
            feature, condition, values = self.parse_rule(rule)
            print(feature, condition, values)
            if condition == "80-between":
                mask = (cluster_data[feature] >= values[0]) & (cluster_data[feature] <= values[1])
            elif condition == "contains":
                mask = cluster_data[feature].isin(values)
            covered_samples.update(cluster_data[mask].index)
        
        coverage = len(covered_samples) / len(cluster_data)
        coverage = len(covered_samples) / len(cluster_data)
        return round(coverage, 4)


    def calculate_separation_error(self, rules, data, labels, cluster):
        """
        Calcula o separation error para um cluster específico.
        
        Parameters:
        - rules: Lista de regras, onde cada posição representa as regras de um cluster.
        - data: DataFrame com os dados.
        - labels: Array com os rótulos dos clusters.
        - cluster: O índice do cluster para o qual calcular o separation error.
        
        Returns:
        - separation_error: Proporção de amostras cobertas pela regra fora do cluster em relação ao total coberto pela regra.
        """
    
        cluster_rules = rules[cluster]
        cluster_data = data[labels == cluster]
        non_cluster_data = data[labels != cluster]
        
        if cluster_data.empty or non_cluster_data.empty:
            return 0.0  
        
        total_covered_samples = set()
        non_cluster_covered_samples = set()
        
        for rule in cluster_rules:
            feature, condition, values = self.parse_rule(rule)
        
            if condition == "80-between":
                cluster_mask = (cluster_data[feature] >= values[0]) & (cluster_data[feature] <= values[1])
                non_cluster_mask = (non_cluster_data[feature] >= values[0]) & (non_cluster_data[feature] <= values[1])
            elif condition == "contains":
                cluster_mask = cluster_data[feature].isin(values)
                non_cluster_mask = non_cluster_data[feature].isin(values)
    
            total_covered_samples.update(cluster_data[cluster_mask].index)
            total_covered_samples.update(non_cluster_data[non_cluster_mask].index)
            non_cluster_covered_samples.update(non_cluster_data[non_cluster_mask].index)
        
        if not total_covered_samples:
            return 0.0
        
        separation_error = len(non_cluster_covered_samples) / len(total_covered_samples)
        return round(separation_error, 4)


    def calculate_conciseness(self, rules, cluster):
        """
        Calcula a conciseness para um cluster específico.
        
        Parameters:
        - rules: Lista de regras, onde cada posição representa as regras de um cluster.
        - cluster: O índice do cluster para o qual calcular a conciseness.
        
        Returns:
        - conciseness: O inverso do número de predicados na explicação. Quanto maior, mais concisa.
        """
    
        cluster_rules = rules[cluster]
        num_predicates = len(cluster_rules)
        
        if num_predicates == 0:
            return 1.0  

        conciseness = 1 / num_predicates
        return round(conciseness, 4)
