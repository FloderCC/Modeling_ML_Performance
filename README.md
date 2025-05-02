# Modeling ML Performance


[![DOI](To be defined)

### Repository of the work entitled "Modeling and Predicting Machine Learning Performance"

## Structure

This repository has the following structure:
```
├── src/datasets
├── src/plots
├── src/results
├── src/results_analysis
├── src/dataset_utils.py
├── src/exp_setup.py
├── src/exp_stage_create_sr_d.py
├── src/exp_stage_create_sr_d_f.py
├── src/exp_stage_create_sr_m.py
├── src/exp_stage_create_sr_mt.py
├── src/exp_stage_dataset_description.py
├── src/exp_stage_meta_dataset_creation.py
├── src/exp_stage_ml_evaluation.py
├── src/exp_stage_ml_tuning.py
├── src/model_utils.py
```

- `src/datasets/` contains the source datasets used in the experiments.  
- `src/plots/` contains all generated plots.  
- `src/results/` contains the results from each experiment stage and the final meta-dataset.  
- `src/results_analysis/` contains the script for generating the plots.  
- `src/dataset_utils.py` contains functions for preprocessing and describing the datasets.  
- `src/exp_setup.py` defines the experiment setup.  
- `src/exp_stage_create_sr_d.py` creates the symbolic regression function based on dataset meta-features.  
- `src/exp_stage_create_sr_d_f.py` creates a symbolic regression function using dataset meta-features, considering only instances with MCC greater than 0.  
- `src/exp_stage_create_sr_m.py` creates the symbolic regression function based on dataset meta-features and model.  
- `src/exp_stage_create_sr_mt.py` creates the symbolic regression function based on dataset meta-features and model type.  
- `src/exp_stage_dataset_description.py` describes the datasets using the `pymfe` library.  
- `src/exp_stage_meta_dataset_creation.py` merges the model evaluation results with the dataset descriptors to create the meta-dataset.  
- `src/exp_stage_ml_evaluation.py` evaluates the ML models using the best hyperparameters obtained in the previous step.  
- `src/exp_stage_ml_tuning.py` performs hyperparameter tuning for the ML models and saves the best hyperparameters.  
- `src/model_utils.py` contains utility methods for creating and evaluating ML models.

## Experiment Stages Order
1. `exp_stage_ml_tuning.py`
2. `exp_stage_ml_evaluation.py`
3. `exp_stage_dataset_description.py`
4. `exp_stage_meta_dataset_creation.py`
5. `exp_stage_create_sr_d.py`
6. `exp_stage_create_sr_d_f.py`
7. `exp_stage_create_sr_m.py`
8. `exp_stage_create_sr_mt.py`

The results from stages 1 to 4 are saved in the `src/results/` folder.  
Stages 5 to 8 are optional and can be executed in any order.

## Datasets used for ML model training

| Index | Dataset Name                                   | Class Name                  | Source                        |
|-------|-----------------------------------------------|-----------------------------|-------------------------------|
| 0     | BoTNeTIoT-L01                                 | label                       | Kaggle                        |
| 1     | DDOS-ANL                                      | PKT_CLASS                   | Kaggle                        |
| 2     | X-IIoTID                                      | class3                      | Kaggle                        |
| 3     | IoTID20                                       | Label                       | Google Datasets               |
| 4     | 5G_Slicing                                    | Slice Type (Output)         | IEEE DataPort                 |
| 5     | IoT-DNL                                       | normality                   | Kaggle                        |
| 6     | NSL-KDD                                       | class                       | IEEE DataPort                 |
| 7     | RT-IoT2022                                    | Attack_type                 | Kaggle                        |
| 8     | QoS-QoE                                       | StallLabel                  | Scientific publication [URL](https://doi.org/10.1109/ICC.2018.8422609) |
| 9     | DeepSlice                                     | slice Type                  | Kaggle                        |
| 10    | Network Slicing Recognition                   | slice Type                  | Kaggle                        |
| 11    | IoT-APD                                       | label                       | Kaggle                        |
| 12    | ASNM-CDX-2009                                 | label_2                     | IEEE DataPort                 |
| 13    | User Network Activities Classification        | output                      | Scientific publication [URL](http://dx.doi.org/10.1007/978-3-031-51135-6_11) |
| 14    | KPI-KQI                                       | Service                     | Zenodo                        |
| 15    | CDC Diabetes Health Indicators                | Diabetes_binary             | UCI ML Repository             |
| 16    | Rain in Australia                             | RainTomorrow                | Kaggle                        |
| 17    | Airline Passenger Satisfaction                | satisfaction                | Kaggle                        |
| 18    | Secondary Mushroom                            | class                       | UCI ML Repository             |
| 19    | Mushroom                                      | class                       | Kaggle                        |
| 20    | Bank Marketing                                | y                           | Kaggle                        |
| 21    | NATICUSdroid                                  | Result                      | UCI ML Repository             |
| 22    | MAGIC Gamma Telescope                         | class                       | UCI ML Repository             |
| 23    | Pulsar                                        | target_class                | Kaggle                        |
| 24    | World Air Quality Index by City and Coordinates | AQI Category              | Kaggle                        |
| 25    | Eye State                                     | eyeDetection                | Kaggle                        |
| 26    | Body performance                              | class                       | Kaggle                        |
| 27    | Customer Segmentation Classification          | Segmentation                | Kaggle                        |
| 28    | Bank Dataset                                  | Exited                      | Kaggle                        |
| 29    | Car Insurance Data                            | OUTCOME                     | Kaggle                        |
| 30    | Employee Future                               | LeaveOrNot                  | Kaggle                        |
| 31    | TUNADROMD                                     | Label                       | UCI ML Repository             |
| 32    | Predict Students Dropout and Academic Success | Target                      | UCI ML Repository             |
| 33    | Breast Cancer                                 | Status                      | Kaggle                        |
| 34    | Apple Quality                                 | Quality                     | Kaggle                        |
| 35    | Water Quality and Potability                  | Potability                  | Kaggle                        |
| 36    | Gender Recognition by Voice                   | label                       | Kaggle                        |
| 37    | Engineering Placements                        | PlacedOrNot                 | Kaggle                        |
| 38    | Students Performance Dataset                  | GradeClass                  | Kaggle                        |
| 39    | NHANES                                        | age_group                   | UCI ML Repository             |
| 40    | Auction Verification                          | verification.result         | UCI ML Repository             |
| 41    | Car Evaluation                                | class                       | UCI ML Repository             |
| 42    | Pistachio types detection                     | Class                       | Kaggle                        |
| 43    | Depression                                    | depressed                   | Kaggle                        |
| 44    | Student Stress Factors                        | stress_level                | Kaggle                        |
| 45    | Milk Quality Prediction                       | Grade                       | Kaggle                        |
| 46    | Home Loan Approval                            | Loan_Status                 | Kaggle                        |
| 47    | Mammographic Mass                             | Severity                    | UCI ML Repository             |
| 48    | Tic-Tac-Toe Endgame                           | Class                       | UCI ML Repository             |
| 49    | Startup Success Prediction                    | status                      | Kaggle                        |
| 50    | Heart Failure Prediction                      | HeartDisease                | Kaggle                        |
| 51    | Diabetes                                      | Outcome                     | Kaggle                        |
| 52    | Balance Scale                                 | Class Name                  | UCI ML Repository             |
| 53    | Congressional Voting                          | Class Name                  | UCI ML Repository             |
| 54    | Cirrhosis Patient Survival Prediction         | Status                      | UCI ML Repository             |
| 55    | Chronic kidney disease                        | Class                       | Kaggle                        |
| 56    | Social Network Ads                            | Purchased                   | Kaggle                        |
| 57    | Differentiated Thyroid Cancer Recurrence      | Recurred                    | UCI ML Repository             |
| 58    | Dermatology                                   | class                       | Kaggle                        |
| 59    | Disease Symptoms and Patient Profile          | Outcome Variable            | Kaggle                        |
| 60    | Haberman Survival                             | Survival status             | UCI ML Repository             |
| 61    | Heart attack possibility                      | target                      | Kaggle                        |
| 62    | Z-Alizadeh Sani                               | Cath                        | UCI ML Repository             |
| 63    | Iris                                          | Species                     | Kaggle                        |
| 64    | Cryotherapy                                   | Result_of_Treatment         | UCI ML Repository             |

## Hyperparameter space explored for the ML models

| **Model**   | **Hyperparameters**                                                                                                                                                                                                                          |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LR          | solver: {saga}<br>C: {0.1, 1, 10}<br>penalty: {l1, l2, None}<br>class_weight: {None, balanced}                                                                                                                                               |
| Ridge       | alpha: {0.1, 1, 10}<br>class_weight: {None, balanced}                                                                                                                                                                                        |
| SGD         | loss: {hinge, log_loss, squared_hinge}<br>alpha: {0.0001, 0.001, 0.01, 0.1}<br>class_weight: {None, balanced}                                                                                                                                 |
| Perceptron  | penalty: {None, l1, l2, elasticnet}<br>alpha: {0.0001, 0.001, 0.01, 0.1}<br>class_weight: {None, balanced}                                                                                                                                   |
| LinearSVC   | C: {0.1, 1, 10}<br>penalty: {l2}<br>class_weight: {None, balanced}                                                                                                                                                                           |
| DT          | criterion: {gini, entropy, log_loss}<br>max_depth: {None, 10, 15, 20}<br>max_features: {None, sqrt, log2}<br>class_weight: {None, balanced}                                                                                                  |
| ET          | criterion: {gini, entropy, log_loss}<br>max_features: {None, sqrt, log2}<br>max_depth: {None, 10, 15, 20}<br>class_weight: {None, balanced}                                                                                                  |
| RF          | criterion: {gini, entropy, log_loss}<br>max_features: {None, sqrt, log2}<br>class_weight: {None, balanced}<br>n_estimators: {25, 50, 100, 200}<br>max_depth: {None, 10, 15, 20}                                                              |
| ETs         | criterion: {gini, entropy, log_loss}<br>max_features: {None, sqrt, log2}<br>class_weight: {None, balanced}<br>n_estimators: {25, 50, 100, 200}<br>max_depth: {None, 10, 15, 20}                                                              |
| AB          | n_estimators: {25, 50, 100, 200}<br>learning_rate: {0.1, 0.5, 1}                                                                                                                                                                             |
| GB          | criterion: {friedman_mse, squared_error}<br>n_estimators: {50, 100, 200}<br>learning_rate: {0.1, 0.5, 1}<br>max_depth: {None, 10, 15, 20}<br>max_features: {None, sqrt, log2}                                                                |
| Bagging     | n_estimators: {10, 50, 100}<br>max_samples: {1.0}                                                                                                                                                                                            |
| GaussianNB  | var_smoothing: {1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1}                                                                                                                                                                                     |
| BernoulliNB | alpha: {0.1, 1, 10}<br>fit_prior: {True}                                                                                                                                                                                                     |
| MLP         | hidden_layer_sizes: {(50,), (100,)}<br>activation: {logistic, relu}<br>alpha: {0.0001, 0.001, 0.01}                                                                                                                                          |
| DNN         | hidden_layer_sizes: {(16,16,16), (8,16,8), (32,16,8)}<br>activation: {logistic, tanh, relu}<br>solver: {adam, sgd}<br>alpha: {0.0001, 0.001, 0.01}<br>batch_size: {10, 50}<br>max_iter: {100, 500}                                            |
