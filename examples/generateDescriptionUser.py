import os
import sys
import asyncio
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_description.cldes import CLDES
from api.promptGemini import PromptGemini as gemini
from cluster_description.cluster_parser import ClusterParser

async def main_workflow(dataset_loader):
    clusterParset = ClusterParser(dataset_loader)
    output = clusterParset.process_dataset()
    
    gemini_instance = gemini(output)
    await gemini_instance.generate(isSemantic=True)

asyncio.run(main_workflow(load_diabetes))

