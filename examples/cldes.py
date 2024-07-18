import os
import sys

from cluster_description.cldes import CLDES
from cluster_description.clex import CLEX
from sklearn.datasets import load_iris, load_wine
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_blobs


def dados_simulados():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750, centers=centers, cluster_std=0.4, random_state=0
    )


    X = StandardScaler().fit_transform(X)
    kmeans = KMeans(3).fit(X)
    predicted = kmeans.predict(X)
    data = pd.DataFrame(X, columns = ['Column_A','Column_B'])
    data["cluster"] = labels_true

    print(data.head())

    groups = [x for x in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        labels_true, 
                                                        test_size=0.30, 
                                                        random_state=42)

    cldes = CLDES(0.01, 10, kmeans)
    pct_chg, pct_chg_acc = cldes.explain_it(X_train, y_train, X_test, y_test, 1, groups)
    print(pct_chg)
    print(pct_chg_acc)

    for i in range(3):
        print(f"Grupo: {i}")
        cldes.generate_cluster_description(data, i, pct_chg)


    fig, ax = plt.subplots()
    ax = sns.scatterplot(
        X[:, 0],
        X[:, 1],
        hue=predicted,
        ax=ax
    )
    legend1 = ax.legend(loc="upper left", title="Grupos")
    ax.add_artist(legend1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()
    

def dados_porto_seguro():
    df = pd.read_csv("~/Documentos/porto_seguro/data/personas_new.csv")

    with open(os.path.join(os.path.dirname(__file__), "colunas.txt"), "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        columns = [x.replace("\n", "") for x in lines]
    
    
    map_products = {
        product: f"Produto {value}" for value, product in enumerate(df["nom_produto_ajustado"].unique())
    }

    map_profissions = {
        profission: f"Profissao {value}" for value, profission in enumerate(df["profissao_ajustada"].unique())
    }

    map_products_and_profissions = {**map_products, **map_profissions}

    df_aux = df[columns].drop(["score_renda", "score_mobilidade"], axis=1)
    df_aux["renda_media"].fillna(df["renda_media"].mean(), inplace=True)
    df_aux["cluster"] = df["cluster"]
    df_aux = df_aux.rename(map_products_and_profissions, axis=1)

    data = df_aux[df_aux["cluster"] != -1].copy()    

    scaler = StandardScaler()
    df_aux["renda_media"] = scaler.fit_transform(df_aux["renda_media"].values.reshape(-1, 1))
    df_aux["tempo_cliente"] = scaler.fit_transform(df_aux["tempo_cliente"].values.reshape(-1, 1))
    df_aux["idade_cliente"] = scaler.fit_transform(df_aux["idade_cliente"].values.reshape(-1, 1))
    print(df_aux.head())
    

    x = df_aux[df_aux["cluster"] != -1].drop("cluster", axis=1)
    y = df_aux[df_aux["cluster"] != -1]["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=0.20
                                                        )

    groups = [i for i in range(len(X_train.columns))]
    
    cldes = CLDES(0.01, 10, kmeans)
    pct_chg, pct_chg_acc = cldes.explain_it(X_train, y_train, X_test, y_test, 10, groups)
    print("Feature importances: ")
    print(pct_chg)
    print("")

    for i in range(int(df["cluster"].nunique()) - 1):
        print(f"Grupo: {i}")
        cldes.generate_cluster_description(data, i, pct_chg)

    clex = CLEX()

    # geração da arvore 
    df.replace(map_products_and_profissions, inplace=True)
    clex.fit(x, y)
    bin_columns = df["profissao_ajustada"].unique().tolist() + df["nom_produto_ajustado"].unique().tolist()

    # geração das regras
    labels = [0, 1, 2, 3, 4] # labels que se deseja gerar regras
    rules_all_groups = clex.get_rules(bin_columns=bin_columns,
                            label=labels,
                            min_samples=0,
                            mutually_exclusives=[df["profissao_ajustada"].unique(), df["nom_produto_ajustado"].unique()] ,
                            )
    #for idx, rules in enumerate(rules_all_groups):     
    #    print(f"Regras do Grupo {labels[idx]}")
    #    for rule in rules:
    #        print(rule)
    #        print("")


    fig = plt.figure(figsize=(40,40))
    ax = plot_tree(clex, 
                    filled=False, 
                    feature_names=data.columns,
                    fontsize=6,
                    label=None,
                    impurity=None)

    plt.show()


if __name__ == "__main__":

    iris = load_iris()
    scaler = MinMaxScaler()
    iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['cluster'])

    X = iris_df.drop("cluster", axis=1)

    for i in X.columns:
        X[i] = scaler.fit_transform(X[i].values.reshape(-1, 1))

    kmeans = KMeans(3).fit(X)
    predicted = kmeans.predict(X)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        iris_df["cluster"],
                                                        test_size=0.20, 
                                                        random_state=42)

    groups = [i for i in range(len(X.columns))]
    cldes = CLDES(0,
                  0)
    
    groups = [i for i in range(len(X.columns))]
    cldes.permutation_feature_importance(X, predicted, groups=groups)
    cldes.get_cluster_description(X, predicted, 0)