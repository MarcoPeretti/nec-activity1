# Load preprocessed data from ./data

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def predict_batch(model, X):
    """
    Run model.predict on each sample in X and stack results as a column vector.

    This is useful for the manual NeuralNet implementation,
    which usually expects a single sample as input to predict().
    """
    return np.array([model.predict(x) for x in X]).reshape(-1, 1)



def mape(y_true, y_pred):
    """
    Safe MAPE implementation (ignores zero targets).

    MAPE = Mean Absolute Percentage Error (in %).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    value = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    return float(value) 




def evaluate_regression(y_true, y_pred):
    """
    Compute MSE, MAE and MAPE for a regression model.
    Metrics are returned in a dictionary.
    """
    mse_val = mean_squared_error(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {
        "MSE":  float(mse_val),
        "MAE":  float(mae_val),
        "MAPE": float(mape_val),
    }



