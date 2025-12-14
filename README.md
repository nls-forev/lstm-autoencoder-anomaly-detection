# LSTM Autoencoder for Time-Series Anomaly Detection

This project implements a sequence-to-sequence LSTM autoencoder
to detect anomalies in univariate time-series data using
reconstruction error.

## Main Features
- Train only on normal behavior
- Compress sequences into a latent state
- Reconstruct input from latent space
- Use reconstruction error as anomaly score

## Model Architecture
- Encoder: multi-layer LSTM
- Latent space: final hidden + cell state
- Decoder: LSTM initialized from encoder state
- Output: per-timestep reconstruction

## Dataset
Household Electric Power Consumption (UCI).
One feature: Global_active_power.

## Results
The model learns normal patterns and produces high reconstruction
error for out-of-distribution windows.

![Reconstruction Error](results/reconstruction_error.png)

## How to Run
1. Download dataset from UCI
2. Create sliding windows
3. Train autoencoder
4. Plot reconstruction error
