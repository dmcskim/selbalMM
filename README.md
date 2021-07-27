# selbalMM
## Selecting Balances with Mixed Models

`selbalMM` is a Python extension to the `selbal` R package. `selbalMM` implements a forward-selection method for identifying groups of taxa whose relative abundance, or balance, is associated with a continuous variable of interest. `selbalMM` uses a mixed model approach, allowing for longitudinal and clustered data.

## Getting Started

### Installation
`selbalMM` can be installed through the python package index PyPI.
```python
pip install selbalMM
```

### Running
`selbalMM` has an object-oriented design modelled after SciKit-Learn. After passing the necessary data, the `fit` method performs cross-validation to determine the optimal number of taxa to include int he balance. The `transform` method calculates the final balance using the full dataset (no holdouts for cross-validation).  A dirichlet sampling procedure is used during the cross-validation step to ensure the optimal number of taxa is not overly influenced by low abundance organisms. 
Note: `X, Y` is microbiome abundance and covariate data, respectively. `LHS/RHS` are the left-hand and right-hand sides of a regression model (patsy format), and `groups` provides the cluster membership.

```python
from selbalMM.selbalMM import selbalMM

model = selbalMM(LHS, RHS, groups)
model.fit(X, Y)
model.transform()
```
