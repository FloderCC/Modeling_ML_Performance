"""
Script to find a symbolic regression model for MCC inference based on dataset features and model type.
"""

import logging
import os
import random
import warnings
import feyn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42):
    # print(f'y = {y}')
    if isinstance(b, str):
        bins = np.histogram_bin_edges(y, bins=b)
        # remove the last index (end point)
        bins = bins[:-1]
    elif isinstance(b, int):
        bins = np.linspace(min(y), max(y), num=b, endpoint=False)
    else:
        raise Exception(f'Undefined bins {b}')

    # print(f'Bins: {bins}')
    groups = np.digitize(y, bins)
    # print(f'Group: {groups}')
    return train_test_split(X, y, test_size=test_size, stratify=groups, random_state=random_state)

# Configure logging
class GreenStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            message = self.format(record)
            green_message = f"\033[92m{message}\033[0m"  # ANSI escape code for green
            self.stream.write(green_message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(processName)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("log_sr_mt.txt", mode='w'),
        GreenStreamHandler()
    ]
)

random_seed = 42

# Set seeds
np.random.seed(random_seed)
random.seed(random_seed)

warnings.filterwarnings("ignore")

df_meta = pd.read_csv('./results/meta_dataset.csv')
# Defining the target column
target_column = 'MCC'

# removing the Model Type column
df_meta = df_meta.drop(columns=['Model'])
# encoding the Model column using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_meta['Model Type'] = le.fit_transform(df_meta['Model Type'])

# Defining the regression score
def smape_score(true, pred):
    return np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2))

# split into train and test sets
X = df_meta.iloc[:, :-1]  # Features
y = df_meta.iloc[:, -1]  # Target variable
# Apply the function
X_train, X_test, y_train, y_test = train_test_split_regression(X, y, test_size=0.2, b='auto')
# Recombine into train/test DataFrames
df_train = pd.DataFrame(X_train, columns=X.columns)
df_train[target_column] = y_train
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test[target_column] = y_test


k = df_train.shape[1] - 1  # Number of predictors

# Instantiate a QLattice
ql = feyn.QLattice(random_seed=random_seed)

# Sample and fit models
models = ql.auto_run(
    data=df_train,
    output_name=target_column,
    n_epochs=50000, # 50000
    criterion=None, # 'bic', 'aic', None
    max_complexity=k * 2, # the maximum amount of features that can be represented is the complexity divided by 2
    function_names=['log', 'exp', 'sqrt', 'squared', 'inverse', 'linear', 'add', 'multiply'],
    threads=os.cpu_count(),
    stypes={col: "f" for col in df_train.columns if col != target_column}
)

# Best model
model = models[0]

y_train = df_train.iloc[:, -1].values  # Target variable

# Predict training set
y_pred_train = model.predict(df_train)
train_r2 = r2_score(y_train, y_pred_train)
train_mape = smape_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)

n = len(y_train)  # Total samples
train_adj_r2 = 1 - (1 - train_r2) * ((n - 1) / (n - k - 1))

# Predict test set
y_test = df_test.iloc[:, -1].values  # Target variable
y_pred_test = model.predict(df_test)
test_r2 = r2_score(y_test, y_pred_test)
test_mape = smape_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

n = len(y_test)
test_adj_r2 = 1 - (1 - test_r2) * ((n - 1) / (n - k - 1))

# Logging results
logging.info(f"Final results:")
logging.info(f"Train dataset ({len(y_train)} rows): R^2: {round(train_r2, 3)}, Adjusted R^2: {round(train_adj_r2, 3)}, sMAPE: {round(train_mape, 3)}, MAE: {round(train_mae, 3)}")
logging.info(f"Test dataset ({len(y_test)} rows): R^2: {round(test_r2, 3)}, Adjusted R^2: {round(test_adj_r2, 3)}, sMAPE: {round(test_mape, 3)}, MAE: {round(test_mae, 3)}")

logging.info("Equation:")
# printing the best model
sympy_model = model.sympify(symbolic_lr=True, include_weights=True)
logging.info(sympy_model.as_expr())