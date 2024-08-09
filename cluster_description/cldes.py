import enum
import logging
from typing import Union
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

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

    def __init__(self, min_per = 0, min_importance = 0, model  = svm.SVC()) -> None:
        self.min_importance = min_importance
        self._min_per = min_per
        self.model = model
        self.pct_chg_recall = None
        self.pct_chg_acc = None
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
            # if sorted_feat_im[i] == 0 or sorted_feat_im[i] < self.min_importance:
            #     continue

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
                lower = float(round(np.percentile(data[col], 10), 3))
                upper = float(round(np.percentile(data[col], 90), 3))
                
                values = f"[{lower}, {upper}]"
                data_to_df[col] = values
                description = Description.continuos_vars(col, lower, upper) \
                    if output_type == OutputType.DESCRIPTION else Predicates.percentile(col, 80, lower, upper)

                generalDescription.append(description)
        return generalDescription, columns_sorted    
