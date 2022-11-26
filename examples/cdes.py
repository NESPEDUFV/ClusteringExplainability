import os
import sys

# adicionar caminho do script
path = "/home/guilherme/Documentos"
sys.path.insert(0, path + "/cluster_explainability")
from cldes import CLDES
from clex import CLEX
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
    cldes.generate_cluster_description(data, 0, pct_chg)


    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels_true
    )
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()
    

def dados_porto_seguro():
    df = pd.read_csv("~/Documentos/porto_seguro/data/personas_new.csv")

    with open(os.path.join(os.path.dirname(__file__), "colunas.txt"), "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
        columns = [x.replace("\n", "") for x in lines]

    df_aux = df[columns].drop(["score_renda", "score_mobilidade"], axis=1)
    df_aux["renda_media"].fillna(df["renda_media"].mean(), inplace=True)
    df_aux["cluster"] = df["cluster"]
    
    data = df_aux[df_aux["cluster"] != -1].copy()    

    scaler = MinMaxScaler()
    df_aux["renda_media"] = scaler.fit_transform(df_aux["renda_media"].values.reshape(-1, 1))
    df_aux["tempo_cliente"] = scaler.fit_transform(df_aux["tempo_cliente"].values.reshape(-1, 1))
    df_aux["idade_cliente"] = scaler.fit_transform(df_aux["idade_cliente"].values.reshape(-1, 1))
    print(df_aux.head())
    

    x = df_aux[df_aux["cluster"] != -1].drop("cluster", axis=1)
    y = df_aux[df_aux["cluster"] != -1]["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=0.20, 
                                                        random_state=42)

    groups = [i for i in range(len(X_train.columns))]
    
    cldes = CLDES(0.01, 10, kmeans)
    pct_chg = cldes.explain_it(X_train, y_train, X_test, y_test, 1, groups)
    print("Feature importances: ")
    print(pct_chg)
    print("")

    cldes.generate_cluster_description(data, 5, pct_chg)

    clex = CLEX()

    # geração da arvore 
    clex.fit(data.drop("cluster", axis=1), data["cluster"])

    # geração das regras
    #labels = [3] # labels que se deseja gerar regras
    #rules_all_groups = clex.get_rules(bin_columns=None,
    #                        label=labels,
    #                        min_samples=0,
    #                        mutually_exclusives=None 
    #                        )

    fig = plt.figure(figsize=(40,40))
    ax = plot_tree(clex, 
                    filled=True, 
                    feature_names=data.columns,
                    fontsize=8)

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

    #dados_simulados()
    #sys.exit()
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            iris_df["cluster"], 
                                                            test_size=0.20, 
                                                            random_state=42)


    groups = [i for i in range(len(X_train.columns))]
    cldes = CLDES(0, 10, kmeans)

    #dados_porto_seguro()
    #sys.exit(0)
    pct_chg, pct_chg_acc = cldes.explain_it(X_train, y_train, X_test, y_test, 1, groups)
    print(iris_df.columns)
    print(pct_chg)


    # ---------------------- g2pc
    #pct_change = cldes.group_permutation_change(X, predicted, 3, groups, 0, 0, 0)

    #print(pct_change)
    for i in range(3):
        cldes.generate_cluster_description(iris_df, i, pct_chg)

    #df = pd.read_csv("/home/guilherme/Documentos/porto_seguro/data/personas_new.csv", index_col=0)
    #print(df.head())

    #print(df.info())

    # ---------------- cluster description

    #cldes.generate_cluster_description(df.drop("cod_pessoa", axis=1), 1)

