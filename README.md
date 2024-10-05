# Distribution-aligned Sequential Counterfactual Explanation with Local Outlier Factor

## Abstract

Sequential counterfactual explanation is one of the counterfactual explanation methods suggesting how to sequentially change the input feature vector to obtain the desired prediction result from a trained classifier.
To show realistic sequential change, existing methods construct a neighborhood graph and obtain a path from the original feature vector to reach a sample for which the model outputs the desired result.
However, constructing an appropriate neighborhood graph is challenging and time-consuming in practice.
This study proposes a new sequential counterfactual explanation method that generates a realistic path without constructing a neighborhood graph.
To evaluate the reality of the suggested path, we first define a cost function based on the Local Outlier Factor (LOF) that assesses how much each vector in the path deviates from the underlying data distribution.
Then, we propose an algorithm for generating a path by iteratively decreasing our cost function.
Since our cost function is non-differentiable due to LOF, we use a local linear approximation to obtain a local descent direction.
Our numerical experiments demonstrated that our method could generate a realistic path that aligns with the data distribution, and its computational time was more stable than the existing method.

## Citation

```
hogehoge
```

## Running the Code


### Artificial data
```bash
python artificial_logistic.py
```

### Real-World Data

```bash
# data: adult
# model: logisitic
python simulate.py setting=adult_logistic

# data: bank
# model: logisitic
python simulate.py setting=bank_logistic

# data: adult
# model: xgboost
python simulate.py setting=adult_xgboost

# data: bank
# model: xgboost
python simulate.py setting=adult_xgboost

# Code to compare preprocessing time
python CE_preprocess_comparison.py
```


