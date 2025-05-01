"""
Script to run the experiment.
"""

import pandas as pd

from src.dataset_utils import dataset_description_header

# opening the dataset with the ML results (results_stage_2)
df_ml_results = pd.read_csv("./results/results_stage_2.csv")

# opening the dataset with the meta-features (results_stage_3)
df_meta_features = pd.read_csv("./results/results_stage_3.csv")

# merging the datasets in order to have the meta-features and the ML results in the same dataset base on Seed, Dataset, and Sample Size
df_meta_dataset = pd.merge(df_ml_results, df_meta_features, on=["Seed", "Dataset", "Sample Size"])


# adding the Model Type column
model_type_dict = {
    # Linear Models
    'LR': 'Linear Model',
    'Ridge': 'Linear Model',
    'SGD': 'Linear Model',
    'Perceptron': 'Linear Model',
    'LinearSVC': 'Linear Model',

    # Tree-Based & Ensemble Methods
    'DT': 'Tree-Based',
    'ExtraTree': 'Tree-Based',
    'ExtraTrees': 'Tree-Based',
    'RF': 'Tree-Based',
    'AdaBoost': 'Tree-Based',
    'GradientBoosting': 'Tree-Based',
    'Bagging': 'Tree-Based',

    # Probabilistic Classifiers
    'GaussianNB': 'Probabilistic Classifier',
    'BernoulliNB': 'Probabilistic Classifier',

    # Neural Networks
    'MLP': 'Neural Network',
    'DNN': 'Neural Network'
}
# Assign model type based on mapping
df_meta_dataset['Model Type'] = df_meta_dataset['Model'].map(model_type_dict)

# keeping only the columns that are going to be used in the study
features_to_select = dataset_description_header + ["Model", "Model Type", "MCC"]
df_meta_dataset = df_meta_dataset[features_to_select]

# saving the meta-dataset
df_meta_dataset.to_csv("./results/meta_dataset.csv", index=False)



