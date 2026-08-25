# Clustering Explainability

Este trabalho propõe o CLINT (Cluster INTerpretability), uma abordagem para interpretar resultados de agrupamento com base na geração automática de descrições legíveis por humanos na forma de regras. A metodologia emprega uma estratégia gulosa orientada por uma função objetivo que equilibra cobertura, erro de separação e concisão, permitindo a construção de descrições que sejam simultaneamente representativas, discriminativas e interpretáveis. Como etapa fundamental do processo, introduzimos um método de seleção de atributos baseado na divergência de Jensen–Shannon, capaz de identificar atributos com alto poder discriminativo por meio da comparação de suas distribuições dentro do cluster com a distribuição global do conjunto de dados.

This work proposes CLINT (Cluster INTerpretability), an approach for interpreting clustering results based on the automatic generation of human-readable descriptions in the form of rules. The methodology employs a greedy strategy driven by an objective function that balances coverage, separation error and conciseness, enabling the construction of descriptions that are simultaneously representative, discriminative, and interpretable. As a fundamental step in the process, we introduce a feature selection method based on Jensen–Shannon divergence, capable of identifying attributes with high discriminative power by comparing their distributions within the cluster against the global dataset.

- **CLEX** — extrai regras legíveis de cada cluster a partir de uma árvore de decisão
  (CART) treinada para separar os grupos.
- **CLDES** — descreve cada cluster por um conjunto mínimo de predicados sobre os
  atributos mais importantes, otimizando *coverage* e *separation error*.

As descrições podem ainda ser traduzidas para linguagem natural por um LLM
(`api/promptGemini.py`), e avaliadas pelas métricas de *coverage*, *separation error*
e *conciseness*.

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
source setup.sh   # adiciona a raiz do projeto ao PYTHONPATH
```

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `clint/clint.py` | `CLDES`: importância de atributos, busca de intervalos e métricas |
| `clint/clex.py` | `CLEX`: árvore de decisão e extração de regras |
| `clint/cluster_parser.py` | `ClusterParser`: pipeline clustering → descrição → métricas |
| `clint/predicates.py`, `clint/description.py` | Formatos de saída (predicados formais e texto) |
| `clusterExplainR/` | Cálculo de importância baseado em entropia usado pelo CLDES |
| `api/promptGemini.py` | Prompt e integração com o Gemini |
| `examples/` | Exemplos de uso ponta a ponta |

## CLEX — regras por árvore de decisão

O método `fit` cria a árvore, recebendo um `DataFrame` com os dados usados no
agrupamento e uma série com o cluster de cada instância. Colunas categóricas são
convertidas em dummies automaticamente.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from clint import CLEX

iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['cluster'])

clex = CLEX()
clex.fit(iris_df.drop("cluster", axis=1), iris_df["cluster"])
```

Depois de criada a árvore, `get_rules` gera as regras:

- `bin_columns`: colunas binárias/dummy, tratadas como "É X" / "Não é X";
- `label`: lista com os rótulos dos grupos desejados;
- `min_samples`: porcentagem mínima de amostras para uma regra ser considerada;
- `mutually_exclusives`: lista (ou lista de listas) de atributos mutuamente
  exclusivos, usada para remover condições redundantes das regras.

```python
labels = [1, 2]
rules_all_groups = clex.get_rules(bin_columns=None,
                                  label=labels,
                                  min_samples=0.1,
                                  mutually_exclusives=None)

for idx, rules in enumerate(rules_all_groups):
    print(f"Regras do Grupo {labels[idx]}")
    for rule in rules:
        print(rule, end="\n\n")
```

```
Regras do Grupo 1
petal length (cm) maior que 2.45 &&
petal width (cm) menor ou igual a 1.75 &&
petal length (cm) menor ou igual a 4.95 &&
petal width (cm) menor ou igual a 1.65
Quantidade: 47 - 94.0%

Regras do Grupo 2
petal length (cm) maior que 2.45 &&
petal width (cm) maior que 1.75 &&
petal length (cm) maior que 4.85
Quantidade: 43 - 86.0%
```

A árvore completa pode ser visualizada com `sklearn.tree.plot_tree(clex, ...)`.

## CLDES — descrição por predicados

A descrição acontece em duas etapas: calcular a importância dos atributos e, a partir
dela, montar a descrição de cada cluster.

```python
from clint import CLDES, OutputType

# min_per: porcentagem mínima de um valor discreto
# min_importance: importância mínima de um atributo
# model: classificador usado na permutation feature importance
cldes = CLDES(min_per=0, min_importance=0)

groups = list(range(len(X.columns)))
cldes.permutation_feature_importance(X, predicted, groups=groups)

description, columns_sorted = cldes.get_cluster_description(
    X, predicted, cluster=0, output_type=OutputType.PREDICATES
)
```

`permutation_feature_importance` usa o rótulo do cluster como variável alvo de um
modelo supervisionado e mede a queda de *recall* ao permutar cada grupo de atributos.
`permutation_feature_importance_entropy` é a alternativa baseada em entropia
(`clusterExplainR`), usada por padrão no `ClusterParser`.

`get_cluster_description` percorre os atributos do mais para o menos importante e
adiciona um predicado por vez enquanto a função objetivo
`0.2 * coverage + 0.8 * (1 - separation_error)` melhorar. A saída pode ser
`OutputType.PREDICATES` (formato formal, ex.: `<petal width (cm), 80-between, [0.1, 0.6]>`)
ou `OutputType.DESCRIPTION` (texto).

### Métricas

| Métrica | Significado |
| --- | --- |
| `calculate_coverage` | Proporção dos pontos do cluster descritos pelas regras (e a pureza dos pontos cobertos) |
| `calculate_separation_error` | Proporção dos pontos cobertos que pertencem a outros clusters |
| `calculate_conciseness` | Inverso do número de predicados da explicação |

## Pipeline completo

`ClusterParser` encadeia clustering, descrição, formatação em linguagem natural e
cálculo das métricas:

```python
from clint.cluster_parser import ClusterParser

cluster_parser = ClusterParser(X, model=RandomForestClassifier(random_state=42), n_clusters=3)
output, metrics = cluster_parser.process_dataset()
```

A geração de texto pelo Gemini está desativada por padrão (as chamadas de rede em
`api/promptGemini.py` estão comentadas). Para usá-la, instale `google-generativeai` e
`python-dotenv` e defina a chave em `API_KEY` — nunca no código.

## Exemplos

```bash
python examples/example_clex.py
python examples/example_cldes.py
python examples/generateDescriptionUser.py
```

## Licença

MIT — veja [LICENSE](LICENSE).
