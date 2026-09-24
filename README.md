# AI4I2020 Predictive Maintenance

A machine learning project for predicting machine failure using the AI4I 2020 Predictive Maintenance dataset.

## Project Objective

The objective of this project is to develop a machine learning system that can predict whether a machine is likely to experience a failure based on operating conditions.

## Input Features

The model uses five input features:

1. Air Temperature
2. Process Temperature
3. Rotational Speed
4. Torque
5. Tool Wear

### Target

- Machine Failure

## Machine Learning Models

Three classification algorithms were implemented:

- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree Classifier

## Model Performance

The models were evaluated using the test dataset.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 98.82% | 98.72% | 98.82% | 98.75% |
| SVM | 97.82% | 95.86% | 97.82% | 96.82% |
| Decision Tree | 98.64% | 98.61% | 98.64% | 98.62% |

These values are measurements obtained on the project's test dataset.

## Project Files

```text
AI4I2020-Predictive-Maintenance/
│
├── README.md
├── requirements.txt
├── train_model.py
├── predict.py
├── random_forest_model.pkl
├── svm_model.pkl
└── decision_tree_model.pkl
## Dataset

This project uses the AI4I 2020 Predictive Maintenance dataset.

The original Excel dataset files are not included in this repository. The repository contains the trained machine learning models (`.pkl`) and the Python scripts used for training and prediction.

The training script expects:

- `TRAIN_AI4I2020_PDM_DATASET.xlsx`
- `TEST_AI4I2020_PDM_DATASET.xlsx`
