import re
import pandas as pd
import numpy as np

def calculate_coverage(rules, data, labels, cluster):
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
        feature, condition, values = parse_rule(rule)
        if condition == "80-between":
            mask = (cluster_data[feature] >= values[0]) & (cluster_data[feature] <= values[1])
        elif condition == "contains":
            mask = cluster_data[feature].isin(values)
        covered_samples.update(cluster_data[mask].index)
    
    coverage = len(covered_samples) / len(cluster_data)
    return coverage



def calculate_separation_error(rules, data, labels, cluster):
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
        feature, condition, values = parse_rule(rule)
    
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
    return separation_error


def calculate_conciseness(rules, cluster):
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
    return conciseness


def parse_rule(rule):
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

rules = [['<absences, 80-between, [0.0, 3.0]>'],
         ['<absences, 80-between, [10.0, 30.68]>', '<Medu, contains, 0.373134328358209>'],
         ['<absences, 80-between, [4.0, 9.92]>']]
labels = np.array([0, 1, 2, 0, 2, 1, 0])  
data = pd.DataFrame({
    "absences": [1, 15, 1, 2, 9, 2, 2],
    "Medu": [0.3, 0.37, 0.29, 0.1, 0.22, 0.37, 0.29]
})

coverage = calculate_coverage(rules, data, labels, cluster=1)
separation_error = calculate_separation_error(rules, data, labels, cluster=1)
conciseness = calculate_conciseness(rules, cluster=1)

print("Coverage:", coverage)
print("Separation Error:", separation_error)
print("conciseness:", conciseness)

