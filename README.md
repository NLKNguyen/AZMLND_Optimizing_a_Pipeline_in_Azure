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

The random parameter sampler provides a way for specifying a range of values for the above parameter that can be randomly selected by the HyperDrive for training in order to find the best performing parameters. There are benefits of using this Random Sampling over Grid Sampling and Bayesian Sampling that Azure ML supports. Only Random Sampling allows early termination of low-performance runs and supports discrete as well as continuous hyperparemeters (e.g `choice`, `uniform`, etc.). In contrast, Grid Sampling only supports discrete hyperparameters (`choice`), and Bayesian Sampling supports some methods for discrete and continuous hyperparameters but doesn't allow early termination. Also, Grid Sampling and Bayesian Sampling takes longer due to more search space to explore, but that can still worth it if we have higher budget and learn from the Random Sampling result for a narrower range of parameters to fine tune further. 

The early stopping policy chosen is Bandit Policy that is based on slack factor/slack amount and evaluation interval. The general purpose is to avoid burning the computation resource on the training processes that are unlikely to yield better result. Our choice of Bandit Policy with a small allowable slack provides more aggresive savings than Median Stopping Policy even though we could risk terminating promising jobs, but it's sufficient for our study. We could also use Truncation Selection Policy with a larger truncation percentage to achieve similar savings as well.

The best model has the accuracy of 0.9102 with `inverse of regularization strength` = 1.1132274914068034 and `maximum number of iterations to converge` = 250

## AutoML

Among 31 models that AutoML generated in the allotted time for the classification task, the best performance model is Voting Ensemble, similar to Stack Ensemble, giving the accuracy of 0.9169. These models are combinations of other standalone models.

```
  ITERATION   PIPELINE                                       DURATION      METRIC      BEST
         0   MaxAbsScaler LightGBM                          0:00:33       0.9152    0.9152
         1   MaxAbsScaler XGBoostClassifier                 0:00:40       0.9153    0.9153
         2   MaxAbsScaler RandomForest                      0:00:28       0.8948    0.9153
         3   MaxAbsScaler RandomForest                      0:00:29       0.8880    0.9153
         4   MaxAbsScaler RandomForest                      0:00:29       0.8099    0.9153
         5   MaxAbsScaler RandomForest                      0:00:28       0.7927    0.9153
         6   SparseNormalizer XGBoostClassifier             0:00:50       0.9116    0.9153
         7   MaxAbsScaler GradientBoosting                  0:00:43       0.9037    0.9153
         8   StandardScalerWrapper RandomForest             0:00:33       0.9002    0.9153
         9   MaxAbsScaler LogisticRegression                0:00:33       0.9083    0.9153
        10   MaxAbsScaler LightGBM                          0:00:28       0.8910    0.9153
        11   SparseNormalizer XGBoostClassifier             0:00:44       0.9121    0.9153
        12   MaxAbsScaler ExtremeRandomTrees                0:02:03       0.8880    0.9153
        13   StandardScalerWrapper LightGBM                 0:00:28       0.8880    0.9153
        14   SparseNormalizer XGBoostClassifier             0:01:43       0.9124    0.9153
        15   StandardScalerWrapper ExtremeRandomTrees       0:00:46       0.8880    0.9153
        16   StandardScalerWrapper LightGBM                 0:00:27       0.8880    0.9153
        17   StandardScalerWrapper LightGBM                 0:00:31       0.9036    0.9153
        18   MaxAbsScaler LightGBM                          0:00:38       0.9046    0.9153
        19   SparseNormalizer LightGBM                      0:00:41       0.9145    0.9153
        20   SparseNormalizer XGBoostClassifier             0:00:34       0.9125    0.9153
        21   MaxAbsScaler LightGBM                          0:00:30       0.9067    0.9153
        22   MaxAbsScaler LightGBM                          0:00:34       0.9098    0.9153
        23   StandardScalerWrapper LightGBM                 0:00:39       0.9079    0.9153
        24   SparseNormalizer XGBoostClassifier             0:00:42       0.9142    0.9153
        25   StandardScalerWrapper XGBoostClassifier        0:00:34       0.8880    0.9153
        26   SparseNormalizer XGBoostClassifier             0:09:49       0.9087    0.9153
        27   MaxAbsScaler LightGBM                          0:00:29       0.9084    0.9153
        28   MaxAbsScaler LightGBM                          0:00:40       0.9086    0.9153
        29   VotingEnsemble                                 0:00:40       0.9169    0.9169
        30   StackEnsemble                                  0:00:50       0.9162    0.9169
```

The best Voting Ensemble model uses variations of XGBoostClassifier, LightGBM, and RandomForest model:

1. MaxAbsScaler XGBoostClassifier
2. MaxAbsScaler LightGBM
3. SparseNormalizer LightGBM
4. SparseNormalizer XGBoostClassifier
5. SparseNormalizer XGBoostClassifier
6. MaxAbsScaler RandomForest 

The configuration of this ensemble is as follow. For full details, check the `best_automl_model.joblib` file in the `outputs` directory.

```
prefittedsoftvotingclassifier
{'estimators': ['1', '0', '19', '24', '20', '4'],
 'weights': [0.08333333333333333,
             0.3333333333333333,
             0.16666666666666666,
             0.16666666666666666,
             0.16666666666666666,
             0.08333333333333333]}

1 - maxabsscaler
{'copy': True}

1 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'gamma': 0,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 3,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'binary:logistic',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

0 - maxabsscaler
{'copy': True}

0 - lightgbmclassifier
{'min_data_in_leaf': 20,
 'n_jobs': 1,
 'problem_info': ProblemInfo(
    dataset_samples=32950,
    dataset_features=122,
    dataset_classes=2,
    dataset_num_categorical=0,
    dataset_categoricals=None,
    pipeline_categoricals=None,
    dataset_y_std=None,
    dataset_uid=None,
    subsampling=False,
    task='classification',
    metric=None,
    num_threads=1,
    pipeline_profile='none',
    is_sparse=True,
    runtime_constraints={'mem_in_mb': None, 'wall_time_in_s': 1800, 'total_wall_time_in_s': 31449600, 'cpu_time_in_s': None, 'num_processes': None, 'grace_period_in_s': None},
    constraint_mode=1,
    cost_mode=1,
    training_percent=None,
    num_recommendations=1,
    model_names_whitelisted=None,
    model_names_blacklisted=None,
    kernel='linear',
    subsampling_treatment='linear',
    subsampling_schedule='hyperband_clip',
    cost_mode_param=None,
    iteration_timeout_mode=0,
    iteration_timeout_param=None,
    feature_column_names=None,
    label_column_name=None,
    weight_column_name=None,
    cv_split_column_names=None,
    enable_streaming=None,
    timeseries_param_dict=None,
    gpu_training_param_dict={'processing_unit_type': 'cpu'}
),
 'random_state': None}

19 - sparsenormalizer
{'copy': True, 'norm': 'l1'}

19 - lightgbmclassifier
{'boosting_type': 'gbdt',
 'colsample_bytree': 0.99,
 'learning_rate': 0.05789894736842106,
 'max_bin': 240,
 'max_depth': 10,
 'min_child_weight': 2,
 'min_data_in_leaf': 0.08276034482758622,
 'min_split_gain': 0.21052631578947367,
 'n_estimators': 400,
 'n_jobs': 1,
 'num_leaves': 197,
 'problem_info': ProblemInfo(
    dataset_samples=32950,
    dataset_features=122,
    dataset_classes=2,
    dataset_num_categorical=0,
    dataset_categoricals=None,
    pipeline_categoricals=None,
    dataset_y_std=None,
    dataset_uid=None,
    subsampling=False,
    task='classification',
    metric=None,
    num_threads=1,
    pipeline_profile='none',
    is_sparse=True,
    runtime_constraints={'mem_in_mb': None, 'wall_time_in_s': 960, 'total_wall_time_in_s': 31449600, 'cpu_time_in_s': None, 'num_processes': None, 'grace_period_in_s': None},
    constraint_mode=1,
    cost_mode=1,
    training_percent=None,
    num_recommendations=1,
    model_names_whitelisted=None,
    model_names_blacklisted=None,
    kernel='linear',
    subsampling_treatment='linear',
    subsampling_schedule='hyperband_clip',
    cost_mode_param=None,
    iteration_timeout_mode=0,
    iteration_timeout_param=None,
    feature_column_names=None,
    label_column_name=None,
    weight_column_name=None,
    cv_split_column_names=None,
    enable_streaming=None,
    timeseries_param_dict=None,
    gpu_training_param_dict={'processing_unit_type': 'cpu'}
),
 'random_state': None,
 'reg_alpha': 0.5789473684210527,
 'reg_lambda': 0.21052631578947367,
 'subsample': 0.09947368421052633}

24 - sparsenormalizer
{'copy': True, 'norm': 'l2'}

24 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.5,
 'eta': 0.001,
 'gamma': 0.01,
 'learning_rate': 0.1,
 'max_delta_step': 0,
 'max_depth': 5,
 'max_leaves': 3,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 50,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.3541666666666667,
 'reg_lambda': 1.3541666666666667,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.7,
 'tree_method': 'auto',
 'verbose': -10,
 'verbosity': 0}

20 - sparsenormalizer
{'copy': True, 'norm': 'l2'}

20 - xgboostclassifier
{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 0.7,
 'eta': 0.3,
 'gamma': 0,
 'grow_policy': 'lossguide',
 'learning_rate': 0.1,
 'max_bin': 63,
 'max_delta_step': 0,
 'max_depth': 6,
 'max_leaves': 3,
 'min_child_weight': 1,
 'missing': nan,
 'n_estimators': 100,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'reg:logistic',
 'random_state': 0,
 'reg_alpha': 1.0416666666666667,
 'reg_lambda': 1.5625,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 0.8,
 'tree_method': 'hist',
 'verbose': -10,
 'verbosity': 0}

4 - maxabsscaler
{'copy': True}

4 - randomforestclassifier
{'bootstrap': True,
 'ccp_alpha': 0.0,
 'class_weight': 'balanced',
 'criterion': 'gini',
 'max_depth': None,
 'max_features': 'log2',
 'max_leaf_nodes': None,
 'max_samples': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 0.01,
 'min_samples_split': 0.01,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 25,
 'n_jobs': 1,
 'oob_score': True,
 'random_state': None,
 'verbose': 0,
 'warm_start': False}
```

## Pipeline comparison

Using AutoML with very little user input, it discovered the best performing model, which has a slight better performance than the Logistic Regression model with parameter tuning by HyperDrive. In addition, the HyperDrive approach requires more manual input for the range of parameters for tuning, and user needs to know more about the algorithm to specify a desirable range.

AutoML takes much longer time than using HyperDrive with the provided parameter sampler. That is expected because AutoML created a large amount of models; whereas HyperDrive just retrain the same model with different parameters.

As a result, AutoML can combine multiple models to ensemble a more complex model than yields higher accuracy. However, it is only slighly better than the HyperDrive approach, and the complex model is much harder to explain from human perspective. There are interpretation benefits of a simpler model even if it has worse performance. Luckily, AutoML still generates dozen of standalone models that can be easier to interpret and still have better performance than the best model found with HyperDrive. Therefore, AutoML is clearly beneficial in all fronts except a longer training time which is a very small trade off.

The best model obtained by HyperDrive has the accuracy of **0.9102** using Logicstic Regression modoel with `inverse of regularization strength` = 2.0042503760886294 and `maximum number of iterations to converge` = 300

The best model obtained by AutoML has the accuracy of **0.9169** using Voting Ensemble model that combines 3 different types of models in various configurations as detailed in the AutoML section.

## Future work

Since the labels in the provided dataset are highly imbalanced (89% vs 11%), this can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class. Improving this will surely be beneficial. One way is to find more data of the smaller class, or using synthetic data generation technique is also useful.

Furthermore, increasing experiment timeout and number of cross validation for AutoML would also improve the performance.

## Proof of cluster clean up

![image](https://user-images.githubusercontent.com/4667129/125048185-83520f80-e054-11eb-9ae3-186e48aead9b.png)
