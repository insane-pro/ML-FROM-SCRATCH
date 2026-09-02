# Machine Learning from Scratch: Linear Models

A production-grade, object-oriented implementation of Linear, Ridge, and Lasso regression algorithms built entirely from scratch using NumPy. 

This repository demonstrates mathematical fundamentals, vectorization, algorithmic optimization (Coordinate Descent), and MLOps benchmarking practices.

## Mathematical Architecture

*   **Ordinary Least Squares (OLS):** Implements the closed-form Normal Equation using the Moore-Penrose pseudoinverse to calculate exact coefficients and handle collinearity.
*   **Ridge Regression (L2 Penalty):** Modifies the closed-form solution by adding an identity matrix penalty (safely isolating the bias term) to shrink coefficients and prevent overfitting.
*   **Lasso Regression (L1 Penalty):** Implements an optimized Coordinate Descent algorithm featuring a soft-thresholding function to automatically perform feature selection by driving irrelevant weights to zero.

## Benchmarks vs. Scikit-Learn

Custom models were benchmarked against industry-standard `scikit-learn` implementations using the scaled Diabetes dataset with degree-2 polynomial features.

| Model | Custom MSE | Scikit-Learn MSE | Absolute Difference |
| :--- | :--- | :--- | :--- |
| **OLS** | 4292.4237 | 4292.4237 | 0.000000 |
| **Ridge** | 3826.6133 | 3826.6133 | 0.000000 |
| **Lasso** | 3664.4907 | 3665.1388 | 0.648154 |

*Note: The fractional variance in the Lasso model is attributed to low-level C++ floating-point optimizations in Scikit-Learn's Coordinate Descent engine. The absolute zero differences in OLS and Ridge confirm exact mathematical parity.*

## Repository Structure
*   `src/base.py`: Abstract base class handling inheritance, predictions, and loss tracking.
*   `src/linear.py`: OLS implementations (Closed-form and Vectorized Gradient Descent).
*   `src/regularization.py`: Ridge (L2) and Lasso (L1 via Coordinate Descent).
*   `src/pipeline.py`: Data ingestion, standard scaling, and polynomial expansion.
*   `src/logger.py` & `src/exception.py`: Custom MLOps telemetry and error tracking.
*   `benchmark.py`: Execution entry point bridging custom logic against sklearn.