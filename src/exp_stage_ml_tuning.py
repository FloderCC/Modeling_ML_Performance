"""
Script for model tuning
"""

import gc
import os
import warnings

import pandas as pd
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.model_selection import train_test_split
from src.dataset_utils import get_dataset_sample, preprocess_dataset, limit_dataset_size
from src.exp_setup import dataset_list, seed_list, dataset_percentage_list, model_list, test_size
from src.model_utils import create_models, reset_seed

# Ignore warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
# Ignore warnings of type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Ignore warnings of type ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Ignore warnings of type FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore warnings of type UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

results_header = ['Seed', 'Dataset', 'Sample Size', 'Model', 'Model Parameters']

results = []

# counter
tuning_number = 1
number_of_tunings = len(seed_list) * len(dataset_list) * len(dataset_percentage_list) * len(model_list)

print(f"Number of tunings to be done: {number_of_tunings}")

for seed in seed_list:
    print(f"Seed: {seed}")
    for dataset_setup in dataset_list:
        dataset_name = dataset_setup[0]
        useful_columns = dataset_setup[1]
        class_name = dataset_setup[2]

        # loading the dataset
        dataset_folder = f"./datasets/{dataset_name}"
        full_df = pd.read_csv(
            f"{dataset_folder}/{[file for file in os.listdir(dataset_folder) if file.endswith('.csv')][0]}")

        if len(useful_columns) > 0:
            print(f"Removing columns {useful_columns}")
            full_df.drop(columns=useful_columns, inplace=True)

        # limiting dataset size to 1m rows
        full_df = limit_dataset_size(full_df, class_name, seed)

        print(f"\033[92mStarted execution with dataset {dataset_name} {full_df.shape}\033[0m")

        # for each dataset size
        for dataset_percentage in dataset_percentage_list:

            # splitting the dataset
            df = get_dataset_sample(full_df, seed, dataset_percentage, class_name, test_size)

            print(f"\033[92m\nStarted execution with dataset sample size {dataset_percentage} ({df.shape[0]} rows)\033[0m")

            if df is None:  # because is it's too small for stratify then its discarded
                print(f"\nDiscarded dataset {dataset_name} with seed {seed} and size {dataset_percentage}")
                tuning_number += len(model_list)
            else:
                # codify & prepare the dataset
                # print("Codifying & preparing dataset ...")
                df = preprocess_dataset(df)

                # processing the class_name column with LabelEncoder to ensure labels starts from 0
                from sklearn.preprocessing import LabelEncoder

                df[class_name] = LabelEncoder().fit_transform(df[class_name])

                # splitting features & label
                X = df.drop(dataset_setup[2], axis=1)
                y = df[dataset_setup[2]]

                # encoding Y to make it processable with DNN models
                y_encoded = pd.get_dummies(y)

                # splitting the dataset in train and test
                x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=test_size,
                                                                                    random_state=seed,
                                                                                    stratify=y_encoded)
                # parsing y_test to a multiclass target
                y_test = tf.argmax(y_test_encoded, axis=1).numpy()

                # resetting the seed
                reset_seed(seed)

                print(f"\033[92mStage 1: Tuning the model {tuning_number}/{number_of_tunings}\033[0m")
                tuned_models = create_models(seed, x_train, y_train_encoded)

                for (model_name, model_parameters_desc) in tuned_models:

                    # ['Seed', 'Dataset', 'Sample Size', 'Model', 'Model Parameters', 'Model No. Iterations']
                    results.append([seed, dataset_name, dataset_percentage, model_name, model_parameters_desc])

                    # dumping results for a file
                    results_df = pd.DataFrame(results, index=None, columns=results_header)

                    # Write to csv
                    results_df.to_csv(f"results/results_stage_1.csv", index=False)

                    tuning_number += 1

                # cleaning up
                del x_train, x_test, y_train_encoded, y_test_encoded, tuned_models
                gc.collect()
