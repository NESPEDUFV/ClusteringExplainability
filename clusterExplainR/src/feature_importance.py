import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, differential_entropy
from typing import List, Union
from scipy.stats import gaussian_kde, entropy
from typing import Union
from statistics import geometric_mean
from scipy.stats import entropy, gaussian_kde

def compute_discrete_entropy(probabilities: Union[np.ndarray, List[float]]) -> float:
    """
    Shannon Entropy for discrete data.
    H(X) = - sum p(x) * log2(p(x))
    """
    probabilities = np.array(probabilities)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_categoric_featureimportance_discrete_entropy(
    data_population: pd.DataFrame,
    column_name: str,
    cluster_column_name: str,
    cluster_name: Union[int, str]
) -> float:
    """
    Feature importance for a categorical feature using discrete entropy.
    FE = 1 - (H_cluster / H_population)
    """
    values = data_population[column_name]

    if values.nunique() == 1:
        return 0.0

    # Probabilidades dentro do cluster
    cluster_probs = (
        data_population.loc[data_population[cluster_column_name] == cluster_name, column_name]
        .value_counts()
    ).values

    # Probabilidades na população
    pop_probs = values.value_counts(normalize=True)

    # Cálculo das entropias
    pop_entropy = entropy(pop_probs, base=2)
    cluster_entropy = entropy(cluster_probs, base=2)

    feature_importance = 1 - (cluster_entropy / pop_entropy) if pop_entropy > 0 else 0
    return min(1, max(0, feature_importance))

def calculate_numeric_featureimportance_continious_entropy(
    data_population: pd.DataFrame,
    column_name: str,
    cluster_column_name: str,
    cluster_name: Union[int, str]
) -> float:
    
    # 1. Separar o grupo do cluster e o grupo "resto" (complemento)
    is_cluster = data_population[cluster_column_name] == cluster_name
    
    group_cluster = data_population.loc[is_cluster, column_name].dropna()
    group_others = data_population[column_name].dropna()

    # Verificação mínima de amostras
    if len(group_cluster) < 5 or len(group_others) < 5:
        return 0.0
    bins = np.histogram_bin_edges(data_population[column_name], bins='auto')
    # 2. Aplicar o teste KS de duas amostras
    # statistic (D): a distância máxima entre as distribuições
    # pvalue: a probabilidade de as duas amostras virem da mesma distribuição
    hist_c, bins_c = np.histogram(group_cluster, bins=bins)
    hist_o, bins_o = np.histogram(group_others, bins=bins)
    #statistic, pvalue = ks_2samp(group_cluster, group_others)
    statistic = jensenshannon(hist_c, hist_o)

    # 3. Usamos o 'statistic' como medida de importância (0 a 1)
    # Opcional: só considerar se o p-value for significativo (ex: < 0.05)
    #if pvalue > 0.05:
    #    return 0.0
    
    return float(statistic)


def _calculate_numeric_featureimportance_continious_entropy(
    data_population: pd.DataFrame,
    column_name: str,
    cluster_column_name: str,
    cluster_name: Union[int, str]
) -> float:
    # Não modifica o DataFrame original
    values = data_population[column_name].dropna()
    
    if values.nunique() <= 1 or len(values) < 10:
        return 0.0
    
    # Normaliza valores
    mean_val = values.mean()
    std_val = values.std()
    
    if std_val == 0:
        return 0.0
    
    population = ((values - mean_val) / std_val).to_numpy()
    
    cluster_values = data_population.loc[
        data_population[cluster_column_name] == cluster_name,
        column_name
    ].dropna()
    
    if len(cluster_values) < 10:
        return 0.0
    
    cluster = ((cluster_values - mean_val) / std_val).to_numpy()
    
    pop_entropy = differential_entropy(population, base=2)
    cluster_entropy = differential_entropy(cluster, base=2)
    
    if pop_entropy <= 0:
        return 0.0
    
    # Considera usar outra métrica, como diferença absoluta normalizada
    feature_importance = 1 - abs(cluster_entropy / pop_entropy)
    
    return float(np.clip(feature_importance, 0.0, 1.0))

""" def calculate_numeric_featureimportance_continious_entropy(
    data_population: pd.DataFrame,
    column_name: str,
    cluster_column_name: str,
    cluster_name
) -> float:

    # Se a feature tiver apenas um valor -> importância 0
    if data_population[column_name].nunique() == 1:
        return 0.0

    # Escalonamento customizado
    col = data_population[column_name]
    col_min, col_max = col.min(), col.max()
    if col_max == col_min:
        data_population[column_name + "_scaled"] = 0
    else:
        data_population[column_name + "_scaled"] = (col - col.mean()) / col.std()

    # População e cluster filtrado
    population = data_population[column_name + "_scaled"].values
    cluster = data_population.loc[
        data_population[cluster_column_name] == cluster_name,
        column_name + "_scaled"
    ].values

    # Estima densidades (equivalente ao density do R)
    kde_pop = gaussian_kde(population)
    kde_cl = gaussian_kde(cluster)

    # Avaliar densidades em pontos 0:100
    x_eval = np.linspace(0, 100, 101)
    pop_density = kde_pop(x_eval)
    cl_density = kde_cl(x_eval)

    # Substituir NaN por 0
    pop_density = np.nan_to_num(pop_density)
    cl_density = np.nan_to_num(cl_density)

    # Entropia (em bits, base=2)
    pop_entropy = entropy(pop_density, base=2)
    cl_entropy = entropy(cl_density, base=2)

    # Cálculo da importância
    if pop_entropy == 0:
        return 0.0
    #print("cluster_colÇ: ", cluster_column_name)
    #print("cluster_entropy: ", cl_entropy)
    #print("pop_entropy: ", pop_entropy)
    feature_importance = 1 - abs(cl_entropy / pop_entropy)

    # Garantir [0,1]
    feature_importance = max(0, min(1, feature_importance))

    return feature_importance """

def calculate_feature_importance_cluster(
    data_clustered: pd.DataFrame,
    cluster_name: Union[int, str],
    numerical_columns: List[int],
    categorical_columns: List[int],
    cluster_column_name: str = "Cluster"
) -> pd.DataFrame:
    """
    Calculates feature importances for all features of a specific cluster.
    """
    results = []
    for cat in categorical_columns:
        colname = data_clustered.columns[cat]
        fe = calculate_categoric_featureimportance_discrete_entropy(
            data_clustered, colname, cluster_column_name, cluster_name
        )
        results.append((cluster_name, colname, fe))

    for num in numerical_columns:
        colname = data_clustered.columns[num]
        fe = max(
            calculate_numeric_featureimportance_continious_entropy(
                data_clustered, colname, cluster_column_name, cluster_name
            ), -1
            #_calculate_numeric_featureimportance_continious_entropy(
            #    data_clustered, colname, cluster_column_name, cluster_name
            #)
        )
        results.append((cluster_name, colname, fe))

    df_results = pd.DataFrame(results, columns=["Cluster_id", "Column_name", "Importance_score"])
    return df_results.sort_values(by="Importance_score", ascending=False)


def calculate_feature_importance(
    data_clustered: pd.DataFrame,
    numerical_columns: List[int],
    categorical_columns: List[int],
    cluster_column_name: str = "Cluster"
) -> pd.DataFrame:
    """
    Calculates feature importances for all clusters.
    """
    all_results = []
    for cluster in data_clustered[cluster_column_name].unique():
        cluster_res = calculate_feature_importance_cluster(
            data_clustered, cluster, numerical_columns, categorical_columns, cluster_column_name
        )
        all_results.append(cluster_res)

    return pd.concat(all_results, ignore_index=True)
