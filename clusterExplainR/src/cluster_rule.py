from clusterExplainR.src.feature_importance import calculate_feature_importance_cluster

import pandas as pd
import itertools
import numpy as np

# --------------------
# Categorical Rule
# --------------------
def categoric_feature_rule(column_name, values):
    def apply(df):
        return df[df[column_name].isin(values)]

    rule = {
        "type": "Rule for a categoric feature",
        "column": column_name,
        "values": values,
        "apply": apply,
        "verbalize": f"{column_name} is " + " or ".join(map(str, values))
    }
    return rule


# --------------------
# Numeric Rule
# --------------------
def numeric_feature_rule(column_name, min_val, max_val):
    def apply(df):
        return df[(df[column_name] >= min_val) & (df[column_name] <= max_val)]

    rule = {
        "type": "Rule for a numeric feature",
        "column": column_name,
        "min": min_val,
        "max": max_val,
        "apply": apply,
        "verbalize": f"{column_name} is between {min_val} and {max_val}"
    }
    return rule


# --------------------
# Conjunction of Rules
# --------------------
def chain_rules_conjunction(rules):
    def apply(df):
        d = df.copy()
        for rule in rules:
            d = rule["apply"](d)
        return d

    rule = {
        "type": "Conjunction of Rules",
        "rules": rules,
        "apply": apply,
        "verbalize": "\n AND ".join([r["verbalize"] for r in rules])
    }
    return rule


# --------------------
# Disjunction of Rules
# --------------------
def chain_rules_disjunction(rules):
    def apply(df):
        dfs = [rule["apply"](df) for rule in rules]
        return pd.concat(dfs).drop_duplicates().reset_index(drop=True)

    rule = {
        "type": "Disjunction of Rules",
        "rules": rules,
        "apply": apply,
        "verbalize": "\n OR ".join([r["verbalize"] for r in rules])
    }
    return rule


# --------------------
# Evaluate Conjunction of Rules
# --------------------
def evaluate_rule_application_in_conjunction(data, rules, target_column, target):
    cp = data.copy()
    total_targets = len(cp[cp[target_column] == target])

    results = []
    for rule in rules:
        before_count = len(cp)
        N = len(cp[cp[target_column] != target])
        P = len(cp[cp[target_column] == target])

        cp = rule["apply"](cp)

        TP = len(cp[cp[target_column] == target])
        coverage = TP / total_targets if total_targets > 0 else 0
        TN = N - len(cp[cp[target_column] != target])
        accuracy = (TP + TN) / before_count if before_count > 0 else 0

        results.append({
            "Rule": rule["verbalize"],
            "Accuracy": accuracy,
            "Coverage": coverage,
            "TP": TP,
            "FP": P - TP,
            "TN": TN,
            "FN": N - TN
        })

        if accuracy >= 1.0:
            break

    return pd.DataFrame(results)


# --------------------
# Rule Generation based on F1
# --------------------
def generate_rule_based_on_F1(data, clustering, categoric_columns, column_name, cluster_name):
    cp = data.copy()
    cp["Cluster"] = clustering

    P = len(cp[cp["Cluster"] == cluster_name])
    N = len(cp[cp["Cluster"] != cluster_name])

    best_rule = None
    best_f1 = 0
    rules = []

    categorical_col_names = [cp.columns[i] for i in categoric_columns]

    if column_name in categorical_col_names:
        cluster_values = cp.loc[cp["Cluster"] == cluster_name, column_name].unique()

        for L in range(1, len(cluster_values) + 1):
            for subset in itertools.combinations(cluster_values, L):
                rules.append(categoric_feature_rule(column_name, list(subset)))
    else:
        cluster_values = cp.loc[cp["Cluster"] == cluster_name, column_name]
        test_ranges = np.linspace(cluster_values.min(), cluster_values.max(), 21)

        for i in range(len(test_ranges) - 1):
            for j in range(i + 1, len(test_ranges)):
                rules.append(numeric_feature_rule(column_name, test_ranges[i], test_ranges[j]))

    for rule in rules:
        rule_result = rule["apply"](cp)
        TP = len(rule_result[rule_result["Cluster"] == cluster_name])
        FP = P - TP
        FN = len(rule_result[rule_result["Cluster"] != cluster_name])
        TN = N - FN

        F1 = TP / (TP + 0.5 * (FP + FN)) if (TP + FP + FN) > 0 else 0

        if F1 > best_f1:
            best_f1 = F1
            best_rule = rule

    return best_rule


# --------------------
# Check if rule is in rule_list
# --------------------
def rule_is_in_rulelist(rule, rule_list):
    return any(r["verbalize"] == rule["verbalize"] for r in rule_list)


# --------------------
# Generate minimal set of rules
# --------------------
def generate_min_set_rules(data, clustering, numerical_columns, categorical_columns, cluster_name):
    cp = data.copy()
    cp["Cluster"] = clustering

    total_targets = len(cp[cp["Cluster"] == cluster_name])
    total_non_targets = len(cp[cp["Cluster"] != cluster_name])

    results = []
    rules = []

    # depende de calculate_feature_importance_cluster (já existe em outro arquivo)
    features = calculate_feature_importance_cluster(cp, cluster_name, numerical_columns, categorical_columns)["Column_name"]
    num_features = len(features)
    used_features = []
    feature = features.iloc[0]

    while len(cp) > 0 and len(used_features) < num_features:
        rule = generate_rule_based_on_F1(cp.drop(columns=["Cluster"]), cp["Cluster"], categorical_columns, feature, cluster_name)
        used_features.append(feature)

        before_count = len(cp)
        N = len(cp[cp["Cluster"] != cluster_name])
        P = len(cp[cp["Cluster"] == cluster_name])

        cp = rule["apply"](cp)

        TP = len(cp[cp["Cluster"] == cluster_name])
        coverage = TP / total_targets if total_targets > 0 else 0
        separation_error = cp[cp["Cluster"] != cluster_name].shape[0]/cp.shape[0]
        TN = N - len(cp[cp["Cluster"] != cluster_name])
        FN = N - TN

        accuracy = (TP + (total_non_targets - FN)) / (total_non_targets + total_targets)

        if TN > 0:
            results.append({
                "Rule": rule["verbalize"],
                "Accuracy": accuracy,
                "Coverage": coverage,
                "separation_error": separation_error,
                "TP": TP,
                "FP": P - TP,
                "TN": TN,
                "FN": FN
            })
            rules.append(rule)

        if FN == 0:
            break

        features = calculate_feature_importance_cluster(cp, cluster_name, numerical_columns, categorical_columns)["Column_name"]
        feature = features.iloc[0]

    df = pd.DataFrame(results)
    single_rule = chain_rules_conjunction(rules)
    verbalization = "\n AND ".join(df["Rule"]) if not df.empty else ""

    if not df.empty:
        last_row = df.iloc[-1]
        verbalization = f"Rule: {cluster_name} ( Acc: {last_row['Accuracy']*100} Cov: {last_row['Coverage']*100}, in %)\n{verbalization}"

    return {"data": df, "rule": single_rule, "verbalization": verbalization, "cluster": cluster_name}
