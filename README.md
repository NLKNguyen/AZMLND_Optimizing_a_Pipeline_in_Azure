# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This dataset contains data about a marketing campaign including people who contacted/answered the calls, and we seek to predict a yes/no answer to whether a person is a potential lead in this campaign.

Two approaches of building the models were in practice:
    1. Build a Logicstic Regression model using Scikit-Learn library with hyperparameter tuning by Azure HyperDrive to find the best model
    2. Build many dozens of models automatically using Azure AutoML to find the best model    

## Scikit-learn Pipeline

The dataset is a CSV file available at a public URL. It contains 20 feature columns and 1 label column for binary classification (yes/no).

The `train.py` program uses Logistic Regression model from Scikit-Learn library for this classification task. 

The model performance metric is accuracy, and its value is logged using the Azure ML SDK for performance comparison during hyperparameter tuning.

The program exposes 2 configurable parameters for the algorithm as program arguments for pararmeter tuning: `inverse of regularization strength` and `maximum number of iterations to converge`.

The random parameter sampler provides a way for specifying a range of values for the above parameter that can be randomly selected by the HyperDrive for training in order to find the best performing parameters.

The early stopping policy chosen is Bandit Policy that is based on slack factor/slack amount and evaluation interval. The general purpose is to avoid burning the computation resource on the training processes that are unlikely to yield better result.

The best model has the accuracy of 0.9101 with `inverse of regularization strength` = 2.0042503760886294 and `maximum number of iterations to converge` = 300

## AutoML

Among over 40 models that AutoML generated in the allotted time for the classification task, the best performance model is Voting Ensemble, similar to Stack Ensemble, giving the accuracy of 0.9169. These models are combinations of other standalone models.


```
 ITERATION   PIPELINE                                       DURATION      METRIC      BEST
         0   MaxAbsScaler LightGBM                          0:00:27       0.9152    0.9152
         1   MaxAbsScaler XGBoostClassifier                 0:00:32       0.9153    0.9153
         2   MaxAbsScaler RandomForest                      0:00:24       0.8920    0.9153
         3   MaxAbsScaler RandomForest                      0:00:25       0.8880    0.9153
         4   MaxAbsScaler RandomForest                      0:00:25       0.8019    0.9153
         5   MaxAbsScaler RandomForest                      0:00:23       0.7765    0.9153
         6   SparseNormalizer XGBoostClassifier             0:00:40       0.9116    0.9153
         7   MaxAbsScaler GradientBoosting                  0:00:35       0.9023    0.9153
         8   StandardScalerWrapper RandomForest             0:00:27       0.9006    0.9153
         9   MaxAbsScaler LogisticRegression                0:00:28       0.9083    0.9153
        10   MaxAbsScaler LightGBM                          0:00:24       0.8910    0.9153
        11   SparseNormalizer XGBoostClassifier             0:00:38       0.9121    0.9153
        12   MaxAbsScaler ExtremeRandomTrees                0:01:38       0.8880    0.9153
        13   StandardScalerWrapper LightGBM                 0:00:25       0.8880    0.9153
        14   SparseNormalizer XGBoostClassifier             0:01:23       0.9124    0.9153
        15   MaxAbsScaler LightGBM                          0:00:28       0.9098    0.9153
        16   StandardScalerWrapper LightGBM                 0:00:25       0.8880    0.9153
        17   StandardScalerWrapper ExtremeRandomTrees       0:00:40       0.8880    0.9153
        18   MaxAbsScaler LightGBM                          0:00:26       0.9061    0.9153
        19   StandardScalerWrapper LightGBM                 0:00:35       0.9079    0.9153
        20   MaxAbsScaler LightGBM                          0:00:27       0.8962    0.9153
        21   SparseNormalizer RandomForest                  0:00:32       0.8880    0.9153
        22   SparseNormalizer XGBoostClassifier             0:00:26       0.8989    0.9153
        23   StandardScalerWrapper LightGBM                 0:00:26       0.8918    0.9153
        24   StandardScalerWrapper XGBoostClassifier        0:00:52       0.9078    0.9153
        25   SparseNormalizer XGBoostClassifier             0:00:26       0.8880    0.9153
        26   SparseNormalizer LightGBM                      0:00:30       0.9112    0.9153
        27   StandardScalerWrapper XGBoostClassifier        0:00:35       0.9081    0.9153
        28   SparseNormalizer XGBoostClassifier             0:00:52       0.9137    0.9153
        29   StandardScalerWrapper LightGBM                 0:00:35       0.9049    0.9153
        30   SparseNormalizer XGBoostClassifier             0:00:27       0.8880    0.9153
        31   SparseNormalizer XGBoostClassifier             0:03:45       0.9122    0.9153
        32   MaxAbsScaler GradientBoosting                  0:00:44       0.9031    0.9153
        33   SparseNormalizer XGBoostClassifier             0:00:46       0.9153    0.9153
        34   SparseNormalizer XGBoostClassifier             0:01:30       0.9141    0.9153
        35   TruncatedSVDWrapper XGBoostClassifier          0:00:32       0.8880    0.9153
        36   SparseNormalizer XGBoostClassifier             0:00:31       0.9117    0.9153
        37   SparseNormalizer XGBoostClassifier             0:00:58       0.9128    0.9153
        38   StandardScalerWrapper XGBoostClassifier        0:00:48       0.9150    0.9153
        39   MaxAbsScaler LightGBM                          0:00:36       0.9086    0.9153
        40   StandardScalerWrapper XGBoostClassifier        0:00:31       0.9077    0.9153
        41   VotingEnsemble                                 0:00:42       0.9169    0.9169
        42   StackEnsemble                                  0:00:52       0.9159    0.9169
```

## Pipeline comparison

Using AutoML with very little user input, it discovered the best performing model, which has a slight better performance than the Logistic Regression model with parameter tuning by HyperDrive. In addition, the HyperDrive approach requires more manual input for the range of parameters for tuning, and user needs to know more about the algorithm to specify a desirable range.

AutoML takes much longer time than using HyperDrive with the provided parameter sampler. That is expected because AutoML created a large amount of models; whereas HyperDrive just retrain the same model with different parameters.

As a result, AutoML can combine multiple models to ensemble a more complex model than yields higher accuracy. However, it is only slighly better than the HyperDrive approach, and the complex model is much harder to explain from human perspective. There are interpretation benefits of a simpler model even if it has worse performance. Luckily, AutoML still generates dozen of standalone models that can be easier to interpret and still have better performance than the best model found with HyperDrive. Therefore, AutoML is clearly beneficial in all fronts except a longer training time which is a very small trade off.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Since the labels in the provided dataset are highly imbalanced (89% vs 11%), this can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class. Improving this will surely be beneficial. One way is to find more data of the smaller class, or using synthetic data generation technique is also useful.

Furthermore, increasing experiment timeout and number of cross validation for AutoML would also improve the performance.


## Proof of cluster clean up

```python
# Cluster cleanup
cpu_cluster.delete()
```


