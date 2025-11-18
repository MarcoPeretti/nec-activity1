import numpy as np
from sklearn.linear_model import LinearRegression


class MultipleLinearRegressionSK:
    """
    Simple wrapper for scikit-learn LinearRegression.

    The purpose of this class is to keep the interface consistent
    with the neural network models:
    - fit(X, Z)
    - predict(X)

    All evaluation metrics (MSE, MAE, MAPE, etc.) will be computed
    in the notebook using the same functions for all models.
    """

    def __init__(self, fit_intercept: bool = True):
        """
        Initialize the linear regression model.

        Parameters
        ----------
        fit_intercept : bool
            Whether the model should learn an intercept (bias) term.
            If False, the regression is forced through the origin.
        """
        self.fit_intercept = bool(fit_intercept)

        # Create the scikit-learn LinearRegression model
        self.model = LinearRegression(fit_intercept=self.fit_intercept)

        print("Multiple Linear Regression (scikit-learn) initialized.")
        print("fit_intercept:", self.fit_intercept)

    # --------------------------
    # Training function
    # --------------------------
    def fit(self, X, Z):
        """
        Train the multiple linear regression model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input feature matrix.
        Z : array-like, shape (n_samples,) or (n_samples, 1)
            Target values for regression.
        """
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)

        # Train the internal scikit-learn model
        self.model.fit(X, Z)

    # --------------------------
    # Prediction function
    # --------------------------
    def predict(self, X):
        """
        Compute predictions for new input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New input samples.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        X = np.asarray(X, dtype=np.float64)
        Z_pred = self.model.predict(X)

        # Ensure output is a NumPy array
        return np.asarray(Z_pred)

    # --------------------------
    # Access learned coefficients
    # --------------------------
    @property
    def coef_(self):
        """Return the learned regression coefficients."""
        return self.model.coef_

    @property
    def intercept_(self):
        """Return the learned intercept (bias) term."""
        return self.model.intercept_


# ----------------------------------------------------------
# Documentation and resources used as reference:
#
# - scikit-learn LinearRegression class:
#   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
#
# - scikit-learn User Guide — Linear Models:
#   https://scikit-learn.org/stable/modules/linear_model.html
#
# These sources were used as a base for:
# - understanding the LinearRegression API (fit, predict, coef_, intercept_)
# - confirming expected input/output shapes
# - following standard scikit-learn practices for regression.
# ----------------------------------------------------------
