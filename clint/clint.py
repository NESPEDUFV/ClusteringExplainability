import ast
import enum
import json
import logging
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from clusterExplainR.src.feature_importance import calculate_feature_importance

from .description import Description
from .predicates import Predicates

logger = logging.getLogger(__name__)

# Pesos da função objetivo usada para escolher intervalos e parar a busca de
# predicados: alta cobertura e baixo erro de separação.
ALPHA_COVERAGE = 0.2
BETA_SEPARATION = 0.8


class OutputType(enum.Enum):
    PREDICATES = "predicates. "
    DESCRIPTION = "description"


class CLDES:

    def __init__(self, min_per=0, min_importance=0, model=RandomForestClassifier(random_state=42)) -> None:
        self.min_importance = min_importance
        self._min_per = min_per
        self.model = model
        self.pct_chg_recall = None
        self.pct_chg_acc = None
        self.rules = []
        self.logger = logger

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
        - description: The list of predicates/descriptions selected for the cluster.
        - columns_sorted: The features ordered by importance.
        """
        self.logger.info("Generating %s", output_type)
        description_user, columns_sorted = self._generate_cluster_description(data, labels, cluster, importance_metric, output_type)
        return description_user, columns_sorted

    def permutation_feature_importance_entropy(self, X, Y, groups, n_repeats=5):
        """
        Calculate entropy based feature importance for the given data.

        Parameters:
        - X: Feature matrix.
        - Y: Target vector.
        - groups: Feature groups for permutation.
        - n_repeats: Number of permutations. Default is 5.

        Returns:
        - importances: Importance score of each feature for each cluster.
        """

        df = X.copy()
        df["Cluster"] = Y
        cols = df.drop("Cluster", axis=1)
        numeric_cols = cols.select_dtypes(include=np.number).columns.tolist()
        discrete_threshold = 0.05  # até 5% de valores únicos ou <= 20 valores únicos
        discrete_cols = [
            col for col in numeric_cols
            if cols[col].nunique() <= 20 or cols[col].nunique() / len(cols) < discrete_threshold
        ]

        continuous_cols = [col for col in numeric_cols if col not in discrete_cols]
        categoric_cols = cols.select_dtypes(exclude=np.number).columns.tolist()
        categoric_cols += discrete_cols

        numeric_indices = [df.columns.get_loc(col) for col in continuous_cols]
        categoric_indices = [df.columns.get_loc(col) for col in categoric_cols]
        importances = calculate_feature_importance(df, numeric_indices, categoric_indices)
        columns_to_drop = [col for col in df.columns if "_scaled" in col]
        df.drop(columns_to_drop, axis=1, inplace=True)
        importances = [
            [
                importances[
                    (importances["Column_name"] == str(col))
                    & (importances["Cluster_id"] == cluster)
                ]["Importance_score"].values[0]
                for col in df.drop("Cluster", axis=1).columns
            ]
            for cluster in df["Cluster"].unique()
        ]

        self.pct_chg_recall = importances
        return importances

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

        # preallocate output matrix number of repeats x number of features
        pct_chg = np.zeros((n_repeats, len(np.unique(groups)), len(clusters)))
        pct_chg_acc = np.zeros((n_repeats, len(np.unique(groups))))
        pct_chg_recall = np.zeros((len(np.unique(groups)), len(clusters)))

        y_predicted = self.model.predict(x_test)
        e_original = accuracy_score(y_test, y_predicted)
        r_original = recall_score(y_test, y_predicted, average=None)
        self.logger.info("Initial model accuracy: %s", e_original)
        self.logger.info("Initial model recall: %s", r_original)
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
        return pct_chg_recall, pct_chg_acc

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

        Features are added to the description one at a time, from the most to the
        least important, while the objective function keeps improving.

        Parameters:
        - data: The dataset containing features.
        - labels: The cluster labels for the data points.
        - cluster: The specific cluster for which to generate a description.
        - importance_type: The metric to use for determining feature importance. Default is "recall".
        - output_type: The type of output to generate. Default is OutputType.DESCRIPTION.

        Returns:
        - general_description: The list of predicates/descriptions selected for the cluster.
        - columns_sorted: The features ordered by importance.
        """
        df = data.copy()
        df = df.dropna(axis=1)

        importances = np.array(self.pct_chg_recall) if importance_type == "recall" else self.pct_chg_acc

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cols = data.select_dtypes(numerics).columns
        continuos_vars = [x for x in cols if len(data[x].unique()) > 10]
        feature_importances = list(importances[cluster, :])
        self.logger.info("Feature importance for cluster %s: %s", cluster, feature_importances)
        index_sort = np.argsort(feature_importances)
        index_sort = index_sort[::-1]

        cluster_rows = np.array(labels) == cluster
        data = data[cluster_rows]
        columns_sorted = data.columns[index_sort]
        self.logger.info("Features sorted by importance: %s", columns_sorted)

        general_description = []
        rules = []
        score = 0
        for col in columns_sorted:
            if col not in continuos_vars:
                values = data[col].unique().tolist()
                description = Description.discrete_vars(col, 100, values) \
                    if output_type == OutputType.DESCRIPTION else Predicates.contains(col, values)
            else:
                lower, upper = self.find_optimal_interval(df, col, labels, cluster)
                description = (
                    Description.continuos_vars(col, lower, upper)
                    if output_type == OutputType.DESCRIPTION
                    else Predicates.percentile(col, 80, lower, upper)
                )

            rules_all = [*rules, self.parse_rule(description)]
            current_coverage, purity = self.calculate_coverage(
                rules_all,
                df,
                np.array(labels),
                cluster
            )
            current_separation_error = self.calculate_separation_error(
                rules_all,
                df,
                np.array(labels),
                cluster
            )

            new_score = (ALPHA_COVERAGE * current_coverage +
                         BETA_SEPARATION * (1 - current_separation_error))
            self.logger.info("Evaluating feature %s: coverage=%.4f, separation_error=%.4f, score=%.4f",
                             col, current_coverage, current_separation_error, new_score)
            if new_score <= score:
                break

            general_description.append(description)
            rules.append(self.parse_rule(description))
            score = new_score

        self.rules = general_description
        return general_description, columns_sorted

    def _iqr_interval(self, data: pd.DataFrame, column: str, cluster: int = None):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.9 * IQR
        upper_bound = Q3 + 1.9 * IQR
        return lower_bound, upper_bound

    def find_optimal_interval(self, df, col, labels, cluster_id):
        """
        Search the pair of cluster percentiles that maximizes the objective function.
        """
        cluster_data = df.loc[labels == cluster_id, col].dropna().sort_values().values

        best_lower, best_upper = None, None
        best_score = -1

        # Testamos combinações de percentis do próprio cluster como candidatos
        percentiles = [i for i in range(0, 101, 5)]
        thresholds = np.percentile(cluster_data, percentiles)

        for i in range(len(thresholds)):
            for j in range(i + 1, len(thresholds)):
                low, high = thresholds[i], thresholds[j]

                in_range = (df[col] >= low) & (df[col] <= high)
                is_cluster = (labels == cluster_id)

                coverage = (in_range & is_cluster).sum() / is_cluster.sum()

                # Erro de separação: quantos de OUTROS clusters estão nessa faixa?
                others_in_range = (in_range & ~is_cluster).sum()
                separation_error = others_in_range / in_range.sum() if in_range.sum() > 0 else 1

                score = (ALPHA_COVERAGE * coverage +
                         BETA_SEPARATION * (1 - separation_error))

                if score > best_score:
                    best_score = score
                    best_lower, best_upper = low, high

        return best_lower, best_upper

    def apply_rules(self, row, rules):
        if not rules:
            return False

        results = []
        for rule in rules:
            try:
                results.append(rule(row))
            except Exception:
                self.logger.exception("Error applying rule %s", rule)

        return all(results)

    def parse_rule(self, rule_str):
        """
        Transforma uma string de regra em um predicado.
        """
        parts = rule_str.strip('<>').split(", ", 2)
        if len(parts) == 3:
            attribute, rule_type, values_str = parts
            if rule_type == "contains":
                value = json.loads(values_str)
                return lambda x: x[attribute] in value
            elif rule_type == "80-between":
                lower_bound, upper_bound = ast.literal_eval(values_str)
                return lambda x: lower_bound <= x[attribute] <= upper_bound
            else:
                raise ValueError(f"Tipo de regra não reconhecido: {rule_type}")
        else:
            raise ValueError(f"Formato inesperado na regra: {rule_str}")

    def get_kde_intervals(self, pop_data, cl_data, resolution=200, threshold=1.1):
        """
        Retorna intervalos onde a densidade do cluster supera a da população.
        threshold=1.2 significa que o cluster deve ser 20% mais denso que a pop.
        """
        min_val, max_val = pop_data.min(), pop_data.max()
        x_eval = np.linspace(min_val, max_val, resolution)

        # Bandwidth automático (Scott)
        kde_pop = gaussian_kde(pop_data)(x_eval)
        kde_cl = gaussian_kde(cl_data)(x_eval)

        # O KDE do scipy já normaliza, aqui só evitamos divisão por zero
        kde_pop = np.where(kde_pop == 0, 1e-10, kde_pop)

        # ratio > threshold indica sobre-representação do cluster
        is_higher = (kde_cl / kde_pop) > threshold

        intervals = []
        if not np.any(is_higher):
            return []

        start_idx = None
        for i, val in enumerate(is_higher):
            if val and start_idx is None:
                start_idx = i
            elif not val and start_idx is not None:
                intervals.append((x_eval[start_idx], x_eval[i - 1]))
                start_idx = None

        if start_idx is not None:
            intervals.append((x_eval[start_idx], x_eval[-1]))
        return intervals[0]

    def calculate_coverage(self, rules, X, predicted, cluster):
        """
        Calcula a cobertura e a pureza de um conjunto de regras.
        rules: conjunto de predicados que definem o cluster
        X: dataset original
        predicted: rótulos previstos pelo algoritmo de clustering
        cluster: o cluster específico para o qual a métrica está sendo calculada
        """
        cluster_points = X[predicted == cluster]
        total_in_cluster = len(cluster_points)

        if total_in_cluster == 0:
            return 0

        covered_in_cluster = 0
        for _, row in cluster_points.iterrows():
            if self.apply_rules(row, rules):
                covered_in_cluster += 1

        coverage = covered_in_cluster / total_in_cluster
        covered = np.all([X.apply(rule, axis=1) for rule in rules], axis=0)
        total_covered = np.sum(covered)
        purity = covered_in_cluster / total_covered if total_covered > 0 else 0
        return round(coverage, 4), round(purity, 4)

    def calculate_separation_error(self, rules, X, predicted, cluster):
        """
        Calcula o erro de separação de um conjunto de regras.
        rules: conjunto de predicados que definem o cluster
        X: dataset original
        predicted: rótulos previstos pelo algoritmo de clustering
        cluster: o cluster específico para o qual o erro está sendo calculado
        """

        if not rules:
            return 0
        covered = np.all([X.apply(rule, axis=1) for rule in rules], axis=0)

        covered_outside_cluster = np.sum(covered & (predicted != cluster))

        total_covered = np.sum(covered)

        separation_error = covered_outside_cluster / total_covered if total_covered > 0 else 0

        return round(separation_error, 4)

    def calculate_conciseness(self, rules):
        """
        Calcula a concisão de uma regra com base no número de predicados.
        rules: lista de predicados que definem o cluster (cada predicado é uma regra).
        """

        num_predicates = len(rules)
        conciseness = 1 / num_predicates if num_predicates > 0 else 0
        return round(conciseness, 4)
