# Wine Quality Prediction Project

## Overview

This repository contains the code, data, and report for this project, which aims to predict the quality of white wine based on its physicochemical properties. We also explore whether models trained on white wine data can be used to predict the quality of red wine.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preparing Dataset](#preparing-dataset)
- [Model Fitting and Evaluation](#model-fitting-and-evaluation)
  - [Baseline Models](#baseline-models)
  - [Non-baseline Models](#non-baseline-models)
- [High Dimensional Models](#high-dimensional-models)
- [Our Own Gradient Descent Implementation](#our-own-gradient-descent-implementation)
- [Out-Of-Distribution Generalisation](#out-of-distribution-generalisation)
- [Acknowledgements](#acknowledgements)

## Dataset Description

We used a fascinating dataset containing physicochemical and sensory data for both red and white wines. This dataset includes 11 attributes like acidity, sugar, and alcohol levels. The goal is to predict the wine quality score, which ranges from 3 to 9.

## Exploratory Data Analysis

To get a sense of the data, we plotted various attributes against wine quality. Here are some intriguing insights:
1. **Fixed acidity**: No clear trend with wine quality.
2. **Free Sulfur Dioxide**: Lower levels might correlate with higher wine quality.
3. **Alcohol**: Higher alcohol content seems to correlate with better wine quality.

## Preparing Dataset

We split the white wine data into training (70%) and testing (30%) sets. To ensure our models perform well, we normalized the numeric predictors and set up 5-fold cross-validation for hyperparameter tuning.

## Model Fitting and Evaluation

### Baseline Models

1. **Simple Linear Regression**: Used three predictors (alcohol, pH, and residual sugar). Achieved an RMSE of 0.774 on the test set.
2. **Multiple Linear Regression**: Considered all predictors. Improved performance with an RMSE of 0.742 on the test set.

### Non-baseline Models

1. **Random Forest**: Our star performer with an RMSE of 0.60 on the test set.
2. **Boosted Trees**: Achieved a notable RMSE of 0.645.
3. **Single Decision Tree**: Simpler but less effective with an RMSE of 0.74.
4. **Cubist Model**: Combined decision trees with linear regression, achieving an RMSE of 0.66.
5. **Elastic Net**: Balanced between lasso and ridge regression; RMSE of 0.74.
6. **Support Vector Machine (SVM)**: Achieved an RMSE of 0.70.
7. **Neural Network (MLP)**: A single hidden layer model with an RMSE of 0.737.
8. **Lasso Regression**: Similar to Elastic Net with an RMSE of 0.74.
9. **Generalized Additive Model (GAM)**: Captured non-linear relationships with an RMSE of 0.729.

## High Dimensional Models

We didn't stop at standard models. We ventured into high-dimensional space by creating polynomial transformations and interaction terms. PCA helped us reduce dimensionality. We used Lasso and SVM for this high-dimensional dataset.

- **SVM without PCA**: Best in class among high-dimensional models.
- **Lasso with PCA**: Showed better performance compared to Lasso without PCA.

## Our Own Gradient Descent Implementation

Why settle for built-in algorithms when you can create your own? We implemented a custom stochastic gradient descent algorithm, adding functionalities like momentum, early stopping, and plotting. This helped us deeply understand the optimization process and the model's convergence behavior.

## Out-Of-Distribution Generalisation

To test the generalization of our models, we used the red wine dataset. We observed:
- **Covariate Shift**: Significant differences in predictors' distributions between red and white wines.
- **Concept Shift**: Simpler models like linear regression generalized better than complex models like Random Forest and Cubist.

## Acknowledgements

We extend our gratitude to the developers of the R packages such as `tidymodels`, `Cubist`, `xgboost`, `ranger`, `keras`, and others that made this project possible. Special thanks to our team for their hard work and collaboration.

For a deeper dive, check out the full report included in the repository. Cheers to better wine predictions! 🍷
