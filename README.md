# Hybrid Spatiotemporal Energy Consumption Forecasting Model

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Data](#data)
   3.1. [Data Generation](#data-generation)
   3.2. [Data Loading](#data-loading)
   3.3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Evaluation and Visualization](#evaluation-and-visualization)
7. [Usage Guide](#usage-guide)
8. [Future Improvements](#future-improvements)

## 1. Project Overview

This project implements a hybrid model for forecasting energy consumption using both temporal and spatial data. The model combines Long Short-Term Memory (LSTM) networks for processing time-series data and Convolutional Neural Networks (CNNs) for handling spatial information. This approach allows the model to capture both time-dependent patterns and geographical influences on energy consumption.

## 2. Dependencies

The project relies on the following Python libraries:
- numpy
- pandas
- tensorflow
- scikit-learn
- matplotlib

Ensure these libraries are installed in your Python environment before running the code.

## 3. Data

### 3.1. Data Generation

The project includes functions to generate synthetic data if real data is not available:

- `generate_synthetic_temporal_data(num_days=365, noise_level=0.1)`: Creates time-series data for energy consumption.
- `generate_synthetic_spatial_data(num_locations=100)`: Generates spatial data for energy consumption across different locations.

### 3.2. Data Loading

The `load_data(temporal_data_path, spatial_data_path)` function attempts to load data from CSV files. If the files are not found, it generates synthetic data and saves it for future use.

### 3.3. Data Preprocessing

- Temporal data preprocessing:
  - `preprocess_temporal_data(data, sequence_length)`: Scales the data and creates sequences for LSTM input.
- Spatial data preprocessing:
  - `preprocess_spatial_data(data, grid_size)`: Converts spatial data into a 2D grid for CNN input.

## 4. Model Architecture

The hybrid model consists of two main components:

1. Temporal Component:
   - LSTM layers for processing time-series data
2. Spatial Component:
   - CNN layers for processing spatial data

These components are combined using a concatenation layer, followed by a dense layer for final predictions.

The model is created using the `create_hybrid_model(temporal_input_shape, spatial_input_shape)` function.

## 5. Training Process

The model is trained using the `train_model(model, X_temporal, X_spatial, y, epochs=50, batch_size=32)` function. Key aspects of the training process include:

- Adam optimizer with a learning rate of 0.001
- Mean Squared Error (MSE) as the loss function
- 50 epochs by default
- 20% of the data is used for validation

## 6. Evaluation and Visualization

The project includes functions for evaluating and visualizing the model's performance:

- `plot_results(history, y_true, y_pred)`: Plots the training history and a comparison of predicted vs. actual values.
- `save_predictions(y_true, y_pred, file_path)`: Saves the predictions to a CSV file for further analysis.
- Root Mean Square Error (RMSE) is calculated to quantify the model's accuracy.

## 7. Usage Guide

To use this model:

1. Prepare your temporal and spatial energy consumption data in CSV format.
2. Update the file paths in the `load_data()` function call.
3. Adjust the `sequence_length` and `grid_size` parameters if necessary.
4. Run the script to train the model and generate predictions.
5. Review the visualizations and RMSE to assess the model's performance.
6. Find the saved predictions in 'predictions.csv' for further analysis.

## 8. Future Improvements

Potential enhancements for the project:

1. Implement data augmentation techniques to improve model generalization.
2. Experiment with different model architectures, such as attention mechanisms or transformer models.
3. Add feature engineering to incorporate external factors (e.g., weather data, holidays).
4. Implement cross-validation for more robust model evaluation.
5. Optimize hyperparameters using techniques like grid search or Bayesian optimization.
6. Develop a user interface for easier interaction with the model.
7. Implement model interpretability techniques to understand the factors influencing predictions.
