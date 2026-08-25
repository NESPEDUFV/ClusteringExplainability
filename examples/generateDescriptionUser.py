"""
Pipeline completo: clustering, descrição dos clusters e cálculo das métricas.

`alternative1` roda um modelo escolhido interativamente; `alternative2` compara
todos os classificadores e salva as métricas em CSV.
"""

import logging
import os
import sys

import pandas as pd
<<<<<<< Updated upstream
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes, load_digits, make_moons, make_circles
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
=======
from sklearn.datasets import load_digits
>>>>>>> Stashed changes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from pymfe.mfe import MFE
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clint.cluster_parser import ClusterParser  # noqa: E402


<<<<<<< Updated upstream
def load_dataset_and_normalize(name: str) -> tuple[pd.DataFrame, pd.Series]:
    data = tuple()

    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "diabetes":
        data = load_diabetes()
    elif name == "digits":
        data = load_digits()
    else:
        return None, None

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    return X_normalized, y

def dataset_moons():
    X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
    X = pd.DataFrame(X, columns=['feature1', 'feature2'])
    y = pd.Series(y, name='target')
    return X, y

def dataset_circles():
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    X = pd.DataFrame(X, columns=['feature1', 'feature2'])
    y = pd.Series(y, name='target')
    return X, y

def analyze_class_complexity(X, y):
    mfe = MFE(groups=["complexity"], summary="mean")
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()
    
    mfe.fit(X_np, y_np)
    
    ft_names, ft_values = mfe.extract()
    
    temp_results = pd.Series(data=ft_values, index=ft_names)
    
    metrics_to_show = ["l1.mean", "l2.mean", "l3.mean", "n1", "n2.mean", "n3.mean"]

    return temp_results[metrics_to_show]

def analise_sensibilidade_recall(change, columns, predicted, name_method="SHAP"):
    sns.heatmap(change, annot=True, cmap="coolwarm",
            xticklabels=[f"Cluster {c}" for c in np.unique(predicted)],
            yticklabels=columns)
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.title("Feature Importance - using " + name_method)
    plt.show()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    

def alternative1():
    name = "breast_cancer"
    X, y = load_dataset_and_normalize(name)

    # X, y = dataset_moons()
    # X, y = dataset_circles()

    metricas = analyze_class_complexity(X, y)

    model = SVC(kernel='linear', random_state=42)
    clusterParser = ClusterParser(X, model, y)
    descricoes, metricas_cluster_box, score, shap_importance_per_cluster, global_shap_importance = clusterParser.process_dataset_sharp()
    
    
    # clear_terminal()

    print("Ánalise de Sensibilidade - Mudança no Recall por Feature e Cluster:")
    analise_sensibilidade_recall(shap_importance_per_cluster, X.columns, clusterParser.predicted)  
    print("Resultados do modelo:")
    print(f"{score}\n")
    print("\nMétricas do Dataset:")
    print(metricas)
    print("\nDescrições dos Clusters:")
    for idx, desc in enumerate(descricoes):
        print(f"Cluster {idx}: {desc}")
        print()
    print("Métricas dos Clusters:")
    if(len(metricas_cluster_box) > 0):
        for cluster_id, metrics in metricas_cluster_box.items():
            print(f"Cluster {cluster_id}: {metrics}")

def alternative2():
    name = "breast_cancer"
    X, y = load_dataset_and_normalize(name)
    
    algorithms = [
        SVC(kernel='linear', random_state=42), 
        SVC(kernel='rbf', random_state=42), 
        RandomForestClassifier(random_state=42), 
        LogisticRegression(), 
        KNeighborsClassifier(), 
        DecisionTreeClassifier(random_state=42)
    ]
    algorithm_names = ["SVC linear", "SVC rbf", "RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier"]

    all_metrics = {}
    metricas = analyze_class_complexity(X, y)

    for algorithm, name in zip(algorithms, algorithm_names):
        clusterParser = ClusterParser(X, algorithm, y)   
        descricoes, metricas_cluster_box, score, _, _ = clusterParser.process_dataset_sharp()
        all_metrics[name] = {
            "cluster_descriptions": descricoes,
            "cluster_metrics": metricas_cluster_box,
            "cluster_model_score": score
        }

    md_content = "# Resultados da Análise de Clusters\n\n"
    md_content += "## Métricas do Dataset\n"
    for metrica, valor in metricas.items():
        md_content += f"- **{metrica}**: {valor}\n"

    md_content += "\n## Métricas e Descrições dos Clusters para todos os algoritmos\n"
    for algo_name, results in all_metrics.items():
        md_content += f"\n### Algoritmo: {algo_name}\n"
        md_content += f"**Resultados do modelo:** {results['cluster_model_score']}\n\n"
        md_content += "**Descrições dos Clusters:**\n"
        for idx, desc in enumerate(results["cluster_descriptions"]):
            md_content += f"- Cluster {idx}: {desc}\n"
        md_content += "\n**Métricas dos Clusters:**\n"
        for cluster_id, metrics in results["cluster_metrics"].items():
            md_content += f"- Cluster {cluster_id}:\n"
            for m, v in metrics.items():
                md_content += f"  - {m}: {v}\n"
        md_content += "\n"

    output_path = "resultados_clusters_sharp.md"
    with open(output_path, "w") as f:
        f.write(md_content)

    return output_path

    # ClusterParser.save_results_csv(dataset_name="student_performance", all_metrics=all_metrics, filename="metrics_output_student_performance")
    # print("Métricas de todos os algoritmos salvas no CSV com sucesso.")


async def main_workflow():
    alternative2()


asyncio.run(main_workflow())
=======
def load_and_preprocess_data(dataset_loader):
    data = dataset_loader()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['cluster'] = data.target

    X = df.drop("cluster", axis=1)

    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


def alternative1(dataset_loader):
    X = load_and_preprocess_data(dataset_loader)
    model = ClusterParser.menu()
    cluster_parser = ClusterParser(X, model)
    output, _ = cluster_parser.process_dataset()

    print(output)


def alternative2(dataset_loader, dataset_name="Digits", filename="metrics_output_digits.csv"):
    X = load_and_preprocess_data(dataset_loader)

    algorithms = {
        "SVC": SVC(),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    }

    all_metrics = {}

    for name, algorithm in algorithms.items():
        cluster_parser = ClusterParser(X, model=algorithm)
        _, output_metrics = cluster_parser.process_dataset()

        all_metrics[name] = output_metrics

        print(f"\nMétricas para o algoritmo {name}:\n{output_metrics}")

    ClusterParser.save_results_csv(dataset_name=dataset_name, all_metrics=all_metrics, filename=filename)
    print("Métricas de todos os algoritmos salvas no CSV com sucesso.")
>>>>>>> Stashed changes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    alternative1(load_digits)
