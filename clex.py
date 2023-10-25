from logging.handlers import TimedRotatingFileHandler
import pandas as pd
from pandas.api.types import is_object_dtype
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import re


class CLEX(DecisionTreeClassifier):

    def __init__(self, *, criterion="gini", 
                        splitter="best", 
                        max_depth=None, 
                        min_samples_split=2, 
                        min_samples_leaf=1, 
                        min_weight_fraction_leaf=0,
                        max_features=None, 
                        random_state=None, 
                        max_leaf_nodes=None, 
                        min_impurity_decrease=0, 
                        class_weight=None, 
                        ccp_alpha=0):
        
        super().__init__(
                criterion=criterion, 
                splitter=splitter, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf, 
                min_weight_fraction_leaf=min_weight_fraction_leaf, 
                max_features=max_features,
                random_state=random_state, 
                max_leaf_nodes=max_leaf_nodes, 
                min_impurity_decrease=min_impurity_decrease, 
                class_weight=class_weight, 
                ccp_alpha=ccp_alpha)

        self.data = None
        self.bin_columns = []


            
    def fit(self, x, y):

        self.data = x.copy()
            
        # realiza o onehot
        for i in self.data.columns:

            if(is_object_dtype(self.data[i])):
                self.bin_columns.extend(x[i].unique())
                dummies = pd.get_dummies(self.data[i])
                self.data = self.data.drop(i, axis=1).join(dummies)

        super().fit(self.data, y)
            
        self.model_score = self.score(self.data, y)
        self.data["cluster"] = y




    def get_rules(self, bin_columns=None, label=None, min_samples=0, mutually_exclusives=None, **kwargs):
        """ 
        Cria as regras para cada cluster

        Esse metódo extrai as regras associadas a cada cluster geradas por uma árvore de decisão. \n
        O metódo tenta gerar regras claras e amigáveis, filtrando regras exclusivas. 
        
        Um exemplo simples de utilização pode ser visto abaixo:            
        
        >>> from clex import CLEX
        >>> x = [[1, 1, 1, 1], [2, 2, 2, 2]]
        >>> y = [0, 0]
        >>> clex = CLEX() # add tree pruning params
        >>> clex.fit(x, y)
        >>> clex.get_rules(label=[1])
        ['1 menor ou igual a 1.5 \nQuantidade: 1 - 100.0%']

        
        Parameters
        ----------
        bin_columns : string
            colunas binarias, representando sim ou nao
        label : int or float
            Label do cluster no qual se deseja extrair as regras
        min_samples : float
            Porcentagem minima de amostras em uma regra    
        mutually_exclusives : list or tuple, optional
            lista de atributos mutuamente exclusivos, utilizado para clareza
            das regras.

        
        Returns
        -------
        numpy.ndarray
            Um conjunto de regras para um ou mais clusters.
        """



        # pegar regras para uma unica label ou para todas
        if(label is not None and len(label) == 1):

            label = int(label[0])
            rules = self._get_rules(data=self.data.query(f"cluster == {label}"),
                            bin_columns=bin_columns, 
                            label=label, 
                            min_samples=min_samples,
                            mutually_exclusives=mutually_exclusives, 
                            **kwargs)
            
            return rules

        else:

            rules = []
            labels = label

            for i in labels: 
                data = self.data.query(f"cluster == {i}")
                rules.append(self._get_rules(
                                data=data,
                                bin_columns=bin_columns, 
                                label=int(i), 
                                min_samples=min_samples,
                                mutually_exclusives=mutually_exclusives, 
                                **kwargs)
                )

            return rules

            
    def _get_rules(self, data=None, 
                        bin_columns=None, 
                        label=None, 
                        min_samples=None, 
                        mutually_exclusives=None, 
                        **kwargs):
        
        x_test = data.drop("cluster", axis=1)
        
        node_indicator = self.decision_path(x_test)
        leaf_id = self.apply(x_test)
        
        tree = self.tree_
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        rule_values = []
        rules = []
        
        mutually_exclusives_keys = {}

        if(mutually_exclusives == None or len(kwargs.keys()) > 0):

            for i in kwargs.keys():
                if(mutually_exclusives == None):
                    mutually_exclusives = [kwargs[i]]
                else: 
                    mutually_exclusives.append(kwargs[i])

        if(mutually_exclusives != None):
            if isinstance(mutually_exclusives[0], list) or isinstance(mutually_exclusives[0], np.ndarray):
                for i in range(len(mutually_exclusives)):
                    mutually_exclusives_keys.update(dict.fromkeys(mutually_exclusives[i], i))
            else: 
                mutually_exclusives_keys.update(dict.fromkeys(mutually_exclusives, 0))

        #print(f"Amostras da classe {label}: ", x_test.shape[0])
        #print("")

        for i in range(x_test.shape[0]): 
            node_index = node_indicator.indices[
                node_indicator.indptr[i] : node_indicator.indptr[i + 1]
            ]

            positives = []
            #conditions = f"Regra cluster {label}\n\n"
            conditions = ""
            for node_id in node_index:
                
                if leaf_id[i] == node_id:
                    porc_value = value[node_id][0][label - 1]/np.sum(value[node_id])
                    rule_values.append(value[node_id][0][label - 1])
                    continue

                # Verifica se a variavel da amostra em questão é maio ou menor que o valor
                if x_test.iloc[i, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                is_bin = bin_columns and x_test.columns[feature[node_id]] in bin_columns
                
                # Trata regras de variavéis binarias diferentes
                if(is_bin): 
                    feature_name = x_test.columns[feature[node_id]]
                    if(x_test.iloc[i, feature[node_id]] == 1):    
                        conditions += ("IS {feature_name}&&\n".format(
                                feature_name=feature_name)
                            )

                    else:
                        conditions += ("NOT IS {feature_name}&&\n".format(
                                feature_name=feature_name)
                            )
                
                else:
                    conditions += ("{feature_name} {inequality} {threshold}&&\n".format(
                            inequality=threshold_sign,
                            threshold=round(threshold[node_id], 2),
                            feature_name=x_test.columns[feature[node_id]])
                    )

            if(porc_value >= 0):      
                rules.append(conditions.strip("&&\n")) 

        rules, counts = np.unique(rules, return_counts=True)        
        rules = rules.astype("U500")

        # filtrar regras inuteis
        new_rules = []
        for i, rule in enumerate(rules):
            positives = []
            for condition in rule:
                
                if(condition[:3] == "IS "):
                    feature_name = condition[3:]
                    if(feature_name in mutually_exclusives_keys):
                            index = mutually_exclusives_keys[feature_name]
                            positives.extend(mutually_exclusives[index])

            conditions_filtered  = self._filter_rules(rule, positives)
            new_rules.append(conditions_filtered)
        
        new_rules = np.array(new_rules, dtype="object")
        rules = new_rules
        total = np.sum(counts)

        # filtrar por uma porcentagem minima de amostras na regra
        rules_filter = np.where(counts/total >= min_samples)
        rules = rules[rules_filter]
        counts = counts[rules_filter]

        #adiciona regras do tipo between
        rules = [self._preprocess_rules(rules[i]) for i in range(len(rules))]
        
        for i in range(len(rules)):
            percent = counts[i]/total * 100
            percent = np.around(percent, 2)
            rules[i] = rules[i] + "Quantidade: " + str(counts[i]) + " - " + str(percent) + "%"

        return rules

    def _filter_rules(self, rules, to_remove):
        new_rules = []
        for idx, i in enumerate(rules): 
            if("NOT " == i[:4]):
                feature_name = i[7:]
                if(feature_name in to_remove):
                    continue
                else:
                    new_rules.append(i)
            else: 
                new_rules.append(i)

        return new_rules
    
    def _preprocess_rules(self, rules):
        features = [re.split(r"<=|>", rules[i])[0].strip() for i in range(len(rules))]
        features = np.unique(features)

        dict_features = {feature: [-np.inf, np.inf] for feature in features}
        new_rules = []

        for i in rules:
            if "<=" in i:
                feature = i.split("<=")[0].strip()
                value = i.split("<=")[1].strip()
                dict_features[feature][1] = min(dict_features[feature][1],float(value)) 
            elif ">" in i:
                feature = i.split(">")[0].strip()
                value = i.split(">")[1].strip()
                dict_features[feature][0] = max(dict_features[feature][0],float(value))

        for i in rules:
            if("IS " in i or "NOT IS " in i):
                new_rules.append(i)
        
        for i in dict_features.keys():
            if(dict_features[i][0] != -np.inf and dict_features[i][1] != np.inf):
                new_rules.append(f"{i} between {dict_features[i][0]} and {dict_features[i][1]}")
            elif(dict_features[i][0] != -np.inf):
                new_rules.append(f"{i} > {dict_features[i][0]}")
            elif(dict_features[i][1] != np.inf):
                new_rules.append(f"{i} <= {dict_features[i][1]}")

        return new_rules

    def _concat_rules(self, rules):

        for idx_rules, rule in enumerate(rules):
            rule_string = ""
            for idx, val in enumerate(rule):
                rule_string += val
                if(idx < len(rule) - 1):
                    rule_string += " &&\n"  
            rules[idx_rules] = rule_string

        return rules

    def top_features(self, k):
        importances_sorted = np.argsort(self.feature_importances_)
        features = self.data.columns
        k = min(len(features), k)
        for i in range(k):
            index = importances_sorted[i]
            print(features[index])
