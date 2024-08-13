from sklearn.datasets import load_iris, load_wine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_description.cldes import CLDES
from api.promptGemini import PromptGemini as gemini

async def main():
    gemini_instance = gemini(formatted_output, columns_sorted)
    await gemini_instance.generate() 

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
cldes = CLDES(0.01, 10, kmeans)

cldes.permutation_feature_importance(X, predicted, groups=groups)

clusterDescription = []
for cluster in np.unique(predicted):
    descriptionUser, columns_sorted = cldes.get_cluster_description(data=X, labels=predicted, cluster=cluster, output_type="description")
    clusterDescription.append(descriptionUser)
    
formatted_output = gemini.format_cluster_descriptions(clusterDescription)

asyncio.run(main())