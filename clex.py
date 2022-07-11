import numpy as np
from sklearn.tree import DecisionTreeClassifier


class CLEX(DecisionTreeClassifier):

    def __init__(self, *, criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0, class_weight=None, ccp_alpha=0):
        super().__init__(
                criterion, 
                splitter, 
                max_depth, 
                min_samples_split, 
                min_samples_leaf, 
                min_weight_fraction_leaf, 
                max_features, random_state, 
                max_leaf_nodes, 
                min_impurity_decrease, 
                class_weight, 
                ccp_alpha)


            
    def fit(self, x, y):

        super().fit(x, y)
        print("Modelo Acc: ", self.score(x, y))


    def get_rules(self, x_test=None, bin_columns=None, label=None, mutually_exclusives=None, **kwargs):
        
        """ Create clusters rules

            Esse metódo extrai as regras associadas a cada cluster através de uma árvore de decisão.
            O metódo tenta gerar regras claras e amigáveis, filtrando regras exclusivas. 
            
            A sample usage can be seen with the following:
            .. code-block :: python
                import pyswarms.backend as P
                from pyswarms.swarms.backend import Swarm, VelocityHandler
                my_swarm = P.create_swarm(n_particles, dimensions)
                my_vh = VelocityHandler(strategy="invert")
                for i in range(iters):
                    # Inside the for-loop
                    my_swarm.velocity = compute_velocity(my_swarm, clamp, my_vh, bounds)
            
            Parameters
            ----------
            swarm : pyswarms.backend.swarms.Swarm
                a Swarm instance
            clamp : tuple of floats, optional
                a tuple of size 2 where the first entry is the minimum velocity
                and the second entry is the maximum velocity. It
                sets the limits for velocity clamping.
            vh : pyswarms.backend.handlers.VelocityHandler
                a VelocityHandler object with a specified handling strategy.
                For further information see :mod:`pyswarms.backend.handlers`.
            bounds : tuple of numpy.ndarray or list, optional
                a tuple of size 2 where the first entry is the minimum bound while
                the second entry is the maximum bound. Each array must be of shape
                :code:`(dimensions,)`.
            Returns
            -------
            numpy.ndarray
                Updated velocity matrix
        """

        node_indicator = self.decision_path(x_test)
        leaf_id = self.apply(x_test)
        
        tree = self.tree_
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        rule_values = []

        rules = np.array([])
                

        print(f"Amostras da classe {label}: ", x_test.shape[0])
        print("")

        for i in range(x_test.shape[0]): 
            node_index = node_indicator.indices[
                node_indicator.indptr[i] : node_indicator.indptr[i + 1]
            ]

            conditions = ""
            for node_id in node_index:
                
                if leaf_id[i] == node_id:
                    porc_value = value[node_id][0][label]/np.sum(value[node_id])
                    rule_values.append(value[node_id][0][label])
                    continue

                # Verifica se a variavel da amostra em questão é maio ou menor que o valor
                if x_test.iloc[i, feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "menor ou igual a"
                else:
                    threshold_sign = "maior que"

                is_bin = x_test.columns[feature[node_id]] in bin_columns
                
                # Trata regras de variavéis binarias diferente
                if(is_bin): 
                    if(x_test.iloc[i, feature[node_id]] == 1):    
                        conditions += "É {feature_name} &&\n".format(
                                feature_name=x_test.columns[feature[node_id]])
                    else:
                        conditions += "Não é {feature_name} &&\n".format(
                                feature_name=x_test.columns[feature[node_id]])
                
                else:
                    conditions += "{feature_name} {inequality} {threshold} &&\n".format(
                            inequality=threshold_sign,
                            threshold=threshold[node_id],
                            feature_name=x_test.columns[feature[node_id]])

            
            if(porc_value >= 0.1):

                rules = np.append(rules,
                                    conditions)
                
                
            return np.unique(rules), np.unique(rule_values)

    