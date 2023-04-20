import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

class CLDES:

    def __init__(self, min_per, num_bins = -1, model = None) -> None:
        self._min_per = min_per
        self._num_bins = num_bins
        self.model = model


    def explain_it(self, X, Y_true, X_test, y_test, n_repeats, groups):
        y_test = np.array(y_test)
        clusters = np.unique(Y_true)
        clf = svm.SVC(decision_function_shape='ovo')
        #clf = DecisionTreeClassifier()
        clf.fit(np.array(X), Y_true) 
        

        Pct_Chg = np.zeros((n_repeats, len(np.unique(groups)), len(clusters))) # preallocate output matrix number of repeats x number of features
        Pct_Chg_acc = np.zeros((n_repeats, len(np.unique(groups)))) # preallocate output matrix number of repeats x number of features
        pct_chg_recall = np.zeros((len(np.unique(groups)), len(clusters))) # preallocate output matrix number of repeats x number of features
        
        y_predicted = clf.predict(X_test)
        e_original = accuracy_score(y_test, y_predicted)
        r_original = recall_score(y_test, y_predicted, average=None)
        print("Acc: ", e_original)
        print("------ Inicio da permutação --------")
        for j in np.unique(groups): 
            print("Repeat: ", j)
            for k in range(n_repeats): 
                X_2 = np.copy(X_test)
                X_2[:] = X_test[:] 
                
                Sub_Data = np.random.permutation(X_2[:, np.squeeze(list(groups == j*np.ones_like(groups)))]) # realiza a permutação
                X_2[:, np.squeeze(list(groups == j*np.ones_like(groups)))] = Sub_Data # adiciona a permutaão na matriz
                
                Y_2 = clf.predict(X_2)
                e_new = accuracy_score(y_test, Y_2)
                r_new = recall_score(y_test, Y_2, average=None)
                
                #e_new = precision_score(y_test, Y_2, average=None)

                #e_new = e_new[cluster]
                #e_original_cluster = e_original[cluster]

                # calcular a porcentagem de mudança na precisão
                #print("Pct_change ", e_original)
                #print("Acc nova: ", e_new)
                
                Pct_Chg_acc[k,j] = (e_original - e_new) 
                for g in clusters.astype(int):   
                    pct_chg_recall[j, int(g)] += r_new[g]
                    cluster = y_test == g
                    Pct_Chg[k, j, int(g)] = np.sum(Y_2[cluster] != y_test[cluster])/len(y_test[cluster])

       
        pct_chg_recall = pct_chg_recall/n_repeats
        pct_chg_recall = r_original - pct_chg_recall
        return pct_chg_recall, Pct_Chg_acc

    def group_permutation_change(self, X, Y, n_repeats, groups, random_state, cluster, check_var):
        
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


    def _entropy1(self, labels, base=None, counts=None):
        
        value, counts = np.unique(labels, return_counts=True)
        return entropy(counts)

    def generate_cluster_description(self, data: pd.DataFrame, cluster: int, importances):
        data = data.dropna(axis=1)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cols = data.select_dtypes(numerics).columns
        continuos_vars = [x for x in cols if len(data[x].unique()) > 15 and data[x].dtype and x != "cluster"]
        
        #for col in continuos_vars: 
        #    bins = np.histogram_bin_edges(data[col], bins="fd")
        #    data[col] = pd.cut(data[col], bins)

        feature_importances = []
        for i in range(importances.shape[0]):
            feature_importances.append(importances[i][cluster])

        #print(feature_importances)
        index_sort = np.argsort(feature_importances)
        index_sort = index_sort[::-1]
        sorte_fe_im = sorted(feature_importances, reverse=True)

        data = data[data["cluster"] == cluster]

        print(index_sort)
        columns_sorted = data.columns[index_sort]
        for col, imp in zip(columns_sorted, sorte_fe_im):
            print(col, " - ", imp)

        data_to_df = {}
        for i, col in enumerate(columns_sorted):
            if(sorte_fe_im[i] == 0):
                continue
            if(col not in continuos_vars):
                value_counts = data[col].value_counts()
                percs = value_counts/data.shape[0]
                percs_idx = percs[percs >= self._min_per].index
                percs = percs[percs_idx].values
                values = value_counts[percs_idx].values    
                
                #print("")
                print(f"{col}: {percs[0]}% valores {percs_idx[0]}")
                #print(pd.DataFrame({
                #    "porc.": percs,
                #    "valores": percs_idx
                #}))
                
                data_to_df[col] = value_counts.index[0]
            else:
                #print(col, data[col])
                lower = float(round(np.percentile(data[col], 15), 3))
                upper = float(round(np.percentile(data[col], 85), 3))
                interval = [lower, upper]
                values = pd.IntervalIndex.from_arrays([lower], [upper], closed="both")
                data_to_df[col] = values
                print(f"Values - {col}: {values} - 75%")
                print("")
        
        print(data_to_df)
        
        #print(data.head())