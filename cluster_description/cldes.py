import logging
from typing import Union
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split


PREDICATES = "predicates"
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
                                output_type: Union[PREDICATES, DESCRIPTION]  = DESCRIPTION,
                                importance_metric = "recall"):


        if output_type == DESCRIPTION:
            self.logger.info("Generating %s", DESCRIPTION)
            for cluster in np.unique(labels):
                self._generate_cluster_description(data,
                                                   labels,
                                                   cluster,
                                                   importance_metric)
    
    def get_cluster_description(self,
                                data,
                                labels,
                                cluster,
                                output_type: Union[PREDICATES, DESCRIPTION] = DESCRIPTION,
                                importance_metric = "recall"):
        
        if output_type == DESCRIPTION:
            self.logger.info("Generating %s", DESCRIPTION)
            self._generate_cluster_description(data,
                                                labels,
                                                cluster,
                                                importance_metric)

            

    def permutation_feature_importance(self, X, Y, groups, n_repeats = 5):
        x_train, x_test, y_train, y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.33, 
                                                            random_state=42)

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
                features_to_permute = np.squeeze(list(groups == j*np.ones_like(groups)))
                sub_data = np.random.permutation(x_test_copy[:, features_to_permute])

                # include new permuted features
                x_test_copy[:, features_to_permute] = sub_data

                y_copy = self.model.predict(x_test_copy)
                e_new = accuracy_score(y_test, y_copy)
                r_new = recall_score(y_test, y_copy, average=None)
                pct_chg_acc[k,j] = (e_original - e_new)
                for g in clusters.astype(int):
                    pct_chg_recall[j, int(g)] += r_new[g]
                    cluster = y_test == g
                    pct_chg[k, j, int(g)] = np.sum(y_copy[cluster] != y_test[cluster])/len(y_test[cluster])

        pct_chg_recall = pct_chg_recall/n_repeats
        pct_chg_recall = r_original - pct_chg_recall
        self.pct_chg_recall = pct_chg_recall
        self.pct_chg_acc = pct_chg_acc
        return pct_chg_recall, pct_chg_acc

    def group_permutation_change(self, X, Y, n_repeats, groups, cluster, check_var):
        
        Pct_Chg = np.zeros((n_repeats,len(np.unique(groups)))) # preallocate output matrix number of repeats x number of features
       
        if check_var == 1:
            R = np.zeros((np.shape(X)[0],len(np.unique(groups)))) # preallocate matrix of ratio of cluster label changes to number of repeats
            VarData = np.zeros_like(R) # preallocate matrix of variance of permuted values for each sample and group
            MeanData = np.zeros_like(R)
            
        for j in np.unique(groups): # for j feature groups
            if check_var == 1:
                Record = np.zeros((np.shape(X)[0], np.sum(list(groups == j*np.ones_like(groups))),n_repeats)) # N Samples x N Features per Group x N Repeats
            for k in range(n_repeats): # for k repeats
                np.random.seed(seed=k)
                X_2 = np.copy(X)
                X_2[:] = X[:] # duplicate data array
                
                Sub_Data = np.random.permutation(X_2[:, np.squeeze(list(groups == j*np.ones_like(groups)))]) # shuffle feature
                X_2[:, np.squeeze(list(groups == j*np.ones_like(groups)))] = Sub_Data # add shuffled data to data matrix
                if check_var == 1:
                    Record[:,:,k] = Sub_Data # keep track of values for each permuted feature

                if(j == 1 and k == 0):
                    print(X_2)
                    print(X)

                Y_CLUSTER = Y == cluster
                Y_2 = self.model.predict(X_2)

                Y_ = Y[Y_CLUSTER] # contar a pct de mudança apenas para determinado grupo
                Y_2_ = Y_2[Y_CLUSTER]
                
                #print(np.sum(np.array(Y)!=np.array(Y_2))/len(np.squeeze(Y)))
                Pct_Chg[k,j] = np.sum(np.array(Y_)!=np.array(Y_2_))/len(np.squeeze(Y)) # calculate percent change
                if check_var == 1:
                    R[:,j] += np.squeeze(np.array(Y)!=np.array(Y_2))/n_repeats
                    VarData[:,j] = np.mean(np.var(Record,axis= 2),axis=1)
                    MeanData[:,j] = np.mean(np.mean(Record,axis= 2),axis=1)

                    
        if check_var == 1:
            return(Pct_Chg, R, VarData, MeanData)
        else:
            return(Pct_Chg)

    def _generate_cluster_description(self, data: pd.DataFrame, labels: Union[np.array, pd.Series, list], cluster: int, importance_type: str = "recall"):
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
        for i, col in enumerate(columns_sorted):
            if(sorted_feat_im[i] == 0 or sorted_feat_im[i] < self.min_importance):
                continue

            if col not in continuos_vars:
                value_counts = data[col].value_counts()
                percs = value_counts/data.shape[0]
                percs_idx = percs[percs >= self._min_per].index
                percs = percs[percs_idx].values
                values = value_counts[percs_idx].values

                for value, percentage in zip(percs, values):
                    print(f"{col}: {percentage}% valores {value}")

                data_to_df[col] = value_counts.index[0]
            else:
                lower = float(round(np.percentile(data[col], 10), 3))
                upper = float(round(np.percentile(data[col], 90), 3))

                values = f"[{lower}, {upper}]"
                data_to_df[col] = values
                print(f"{col}: 80% values between {values}")
