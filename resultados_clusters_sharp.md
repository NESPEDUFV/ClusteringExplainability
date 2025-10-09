# Resultados da Análise de Clusters

## Métricas do Dataset
- **l1.mean**: 0.012463794887722202
- **l2.mean**: 0.008787346221441172
- **l3.mean**: 0.010544815465729385
- **n1**: 0.07381370826010544
- **n2.mean**: 0.36106676881267735
- **n3.mean**: 0.0492091388400703

## Métricas e Descrições dos Clusters para todos os algoritmos

### Algoritmo: SVC linear
**Resultados do modelo:**               precision    recall  f1-score   support

           0       0.98      0.97      0.98        67
           1       0.98      0.99      0.99       121

    accuracy                           0.98       188
   macro avg       0.98      0.98      0.98       188
weighted avg       0.98      0.98      0.98       188


**Descrições dos Clusters:**
- Cluster 0: ['<worst concave points, 80-between, [0.25, 0.981]>', '<mean concave points, 80-between, [0.135, 0.932]>']
- Cluster 1: ['<worst concave points, 80-between, [0.0, 0.538]>', '<mean concave points, 80-between, [0.0, 0.359]>']

**Métricas dos Clusters:**
- Cluster 0:
  - coverage: 1.0
  - separation_error: 0.4817
  - conciseness: 0.5
- Cluster 1:
  - coverage: 0.9972
  - separation_error: 0.1927
  - conciseness: 0.5


### Algoritmo: SVC rbf
**Resultados do modelo:**               precision    recall  f1-score   support

           0       0.97      0.97      0.97        67
           1       0.98      0.98      0.98       121

    accuracy                           0.98       188
   macro avg       0.98      0.98      0.98       188
weighted avg       0.98      0.98      0.98       188


**Descrições dos Clusters:**
- Cluster 0: ['<worst concave points, 80-between, [0.25, 0.981]>', '<mean concave points, 80-between, [0.135, 0.932]>']
- Cluster 1: ['<worst concave points, 80-between, [0.0, 0.538]>', '<mean concave points, 80-between, [0.0, 0.359]>']

**Métricas dos Clusters:**
- Cluster 0:
  - coverage: 1.0
  - separation_error: 0.4817
  - conciseness: 0.5
- Cluster 1:
  - coverage: 0.9972
  - separation_error: 0.1927
  - conciseness: 0.5


### Algoritmo: RandomForestClassifier
**Resultados do modelo:**               precision    recall  f1-score   support

           0       0.95      0.93      0.94        67
           1       0.96      0.98      0.97       121

    accuracy                           0.96       188
   macro avg       0.96      0.95      0.95       188
weighted avg       0.96      0.96      0.96       188


**Descrições dos Clusters:**
- Cluster 0: ['<worst concave points, 80-between, [0.25, 0.981]>', '<mean concave points, 80-between, [0.135, 0.932]>', '<worst area, 80-between, [0.091, 0.749]>']
- Cluster 1: ['<worst concave points, 80-between, [0.0, 0.538]>', '<mean concave points, 80-between, [0.0, 0.359]>', '<worst area, 80-between, [0.015, 0.185]>']

**Métricas dos Clusters:**
- Cluster 0:
  - coverage: 1.0
  - separation_error: 0.5401
  - conciseness: 0.3333
- Cluster 1:
  - coverage: 1.0
  - separation_error: 0.2222
  - conciseness: 0.3333


### Algoritmo: LogisticRegression
**Resultados do modelo:**               precision    recall  f1-score   support

           0       0.97      0.93      0.95        67
           1       0.96      0.98      0.97       121

    accuracy                           0.96       188
   macro avg       0.96      0.95      0.96       188
weighted avg       0.96      0.96      0.96       188


**Descrições dos Clusters:**
- Cluster 0: ['<worst concave points, 80-between, [0.25, 0.981]>', '<mean concave points, 80-between, [0.135, 0.932]>']
- Cluster 1: ['<worst concave points, 80-between, [0.0, 0.538]>', '<mean concave points, 80-between, [0.0, 0.359]>']

**Métricas dos Clusters:**
- Cluster 0:
  - coverage: 1.0
  - separation_error: 0.4817
  - conciseness: 0.5
- Cluster 1:
  - coverage: 0.9972
  - separation_error: 0.1927
  - conciseness: 0.5


### Algoritmo: KNeighborsClassifier
**Resultados do modelo:**               precision    recall  f1-score   support

           0       0.96      0.96      0.96        67
           1       0.98      0.98      0.98       121

    accuracy                           0.97       188
   macro avg       0.97      0.97      0.97       188
weighted avg       0.97      0.97      0.97       188


**Descrições dos Clusters:**
- Cluster 0: ['<worst concave points, 80-between, [0.25, 0.981]>']
- Cluster 1: ['<worst concave points, 80-between, [0.0, 0.538]>']

**Métricas dos Clusters:**
- Cluster 0:
  - coverage: 0.9717
  - separation_error: 0.4731
  - conciseness: 1.0
- Cluster 1:
  - coverage: 0.9888
  - separation_error: 0.1514
  - conciseness: 1.0


### Algoritmo: DecisionTreeClassifier
**Resultados do modelo:**               precision    recall  f1-score   support

           0       0.87      0.91      0.89        67
           1       0.95      0.93      0.94       121

    accuracy                           0.92       188
   macro avg       0.91      0.92      0.91       188
weighted avg       0.92      0.92      0.92       188


**Descrições dos Clusters:**
- Cluster 0: ['<mean concave points, 80-between, [0.135, 0.932]>', '<worst concavity, 80-between, [0.099, 0.765]>', '<worst perimeter, 80-between, [0.206, 0.845]>', '<worst radius, 80-between, [0.195, 0.894]>']
- Cluster 1: ['<mean concave points, 80-between, [0.0, 0.359]>', '<worst concavity, 80-between, [0.0, 0.565]>', '<worst perimeter, 80-between, [0.034, 0.324]>', '<worst radius, 80-between, [0.039, 0.344]>']

**Métricas dos Clusters:**
- Cluster 0:
  - coverage: 1.0
  - separation_error: 0.5574
  - conciseness: 0.25
- Cluster 1:
  - coverage: 1.0
  - separation_error: 0.3602
  - conciseness: 0.25

