# Breast Cancer Decision Tree Analysis

This repository contains an analysis of the Breast Cancer Wisconsin dataset using decision tree classifiers. The project explores hyperparameter tuning, visualizes decision trees, and evaluates classifier performance through training and testing accuracies.

## Project Overview
- **Dataset**: Breast Cancer Wisconsin dataset, fetched using Scikit-Learn.
- **Objective**: To classify instances of breast cancer as malignant or benign using decision tree classifiers.
- **Techniques Used**:
  - Decision tree training with entropy as the splitting criterion.
  - Hyperparameter optimization using GridSearchCV.
  - Visualizing tree structures and evaluating accuracy at different depths.

## Features
1. Data preprocessing and splitting into training and testing datasets (60%-40% split).
2. Decision tree classification with:
   - Criterion: Entropy.
   - Minimum samples per split: 6.
3. Training and test accuracy plotted against tree depths.
4. GridSearchCV for optimizing `max_depth` and `min_samples_split` hyperparameters.
5. Visualization of the decision tree for the best model.
