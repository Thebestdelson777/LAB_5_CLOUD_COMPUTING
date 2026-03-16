# Lab 5: Feature Selection, RUL Prediction, and Cloud Deployment

## Overview
This lab demonstrates an end-to-end workflow for predicting **Remaining Useful Life (RUL)** of machines using time-series features.  
The workflow covers:

- Feature extraction and selection
- Test set preparation
- Model loading and prediction
- Saving predictions for further analysis

This lab was performed in an **Azure ML cloud environment**.

## Folder Structure


```text
lab5/
│
├── features/
│   ├── tsfresh_features.csv             # Full extracted features for training
│   ├── tsfresh_features_selected.csv    # Selected features from training
│   ├── selected_feature_list.csv        # List of selected feature names
│   ├── tsfresh_features_test.csv        # Features for the test set
│   └── test_set_predictions_final.csv   # Final RUL predictions for the test set
│
├── models/
│   └── random_forest_rul.pkl           # Trained Random Forest model
│
├── scripts/
│   ├── feature_selection_test.py       # Script for selecting features in the test set
│   ├── predict_test_set.py             # Initial test set prediction script
│   ├── check_model_features.py         # Inspect features expected by the model
│   ├── check_test_features.py          # Inspect features present in the test set
│   ├── predict_test_set_aligned.py     # Script to align test set with model features
│   └── predict_test_set_final.py       # Final working prediction script
│
└── README.md                           # This technical report
```

Steps Performed
1. Load and Inspect Test Features
Loaded the test set features (tsfresh_features_test.csv) using Pandas.
Checked the shape and column names to ensure they are consistent with the training set.

2. Feature Selection
Loaded the selected feature list from selected_feature_list.csv.
Filtered the test set to include only the features used in training.
Any missing features were added with default value 0 to match the model.

3. Model Loading
Loaded the trained Random Forest model (random_forest_rul.pkl) using joblib.
Verified the model expected 9,487 features, same as used during training.

4. Align Test Features with Model
Renamed columns or added missing columns to match the model’s expected input.
Ensured the order of features matches the training phase.
5. Generate Predictions
Predicted RUL for all test samples using the aligned test set.
Saved predictions to features/test_set_predictions_final.csv.

Sample predictions:
RUL_prediction
155.75
153.71
157.87
155.13
156.04
156.7
154.63
155.98
156.79
156.12

Key Observations
Feature alignment is critical: the model will raise a ValueError if test features do not match the training features.
Adding many columns one by one may generate PerformanceWarnings in Pandas; these do not affect predictions.
After alignment, the test set predictions are consistent and ready for evaluation.

Usage
To reproduce test set predictions:
# Step 1: Select features for the test set
python scripts/feature_selection_test.py
# Step 2: Run final prediction script
python scripts/predict_test_set_final.py
# Step 3: Inspect first few predictions
head -n 10 features/test_set_predictions_final.csv

Dependencies
Python 3.10
Pandas
scikit-learn
joblib
Azure ML SDK (if running in Azure cloud)


