import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MultipleLinearRegressionSK:
    """
    Multiple Linear Regression using scikit-learn.

    Simple wrapper that mimics a fit/predict workflow similar to the neural net:
    - Accepts a validation split.
    - Stores train/validation metrics (MSE and R^2).
    - Optionally scales features with StandardScaler.
    """

    def __init__(self, val_split=0.0, fit_intercept=True, scale=False, random_state=42):
        """
        Parameters
        ----------
        val_split : float
            Percentage of data used for validation (between 0 and 1). Example: 0.2 -> 20%
        fit_intercept : bool
            Whether to calculate the intercept for this model.
        scale : bool
            If True, apply StandardScaler to X before LinearRegression.
        random_state : int
            Controls the randomness of the train/validation split (for reproducibility).
        """
        self.val_split = float(val_split)
        self.fit_intercept = bool(fit_intercept)
        self.scale = bool(scale)
        self.random_state = int(random_state)

        # Pipeline: optional scaler + linear regression
        steps = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("linreg", LinearRegression(fit_intercept=self.fit_intercept)))
        self.model = Pipeline(steps)

        # Holders for metrics and split data shapes
        self.train_mse_ = None
        self.train_r2_ = None
        self.val_mse_ = None
        self.val_r2_ = None
        self.n_features_in_ = None
        self.n_targets_ = None

    def fit(self, X, Z):
        """
        Train the Multiple Linear Regression model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.
        Z : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            Targets. Supports single-output or multi-output regression.
        """
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)

        # Keep shapes information
        self.n_features_in_ = X.shape[1]
        self.n_targets_ = 1 if Z.ndim == 1 else Z.shape[1]

        # Train/validation split (if requested)
        if self.val_split > 0.0:
            X_train, X_val, Z_train, Z_val = train_test_split(
                X, Z, test_size=self.val_split, random_state=self.random_state, shuffle=True
            )
        else:
            X_train, Z_train = X, Z
            X_val, Z_val = None, None

        # Fit pipeline
        self.model.fit(X_train, Z_train)

        # Metrics on training
        Z_pred_train = self.model.predict(X_train)
        self.train_mse_ = float(mean_squared_error(Z_train, Z_pred_train))
        self.train_r2_ = float(r2_score(Z_train, Z_pred_train))

        # Metrics on validation (if available)
        if X_val is not None:
            Z_pred_val = self.model.predict(X_val)
            self.val_mse_ = float(mean_squared_error(Z_val, Z_pred_val))
            self.val_r2_ = float(r2_score(Z_val, Z_pred_val))
        else:
            self.val_mse_ = None
            self.val_r2_ = None

        # Print a short summary (useful during the assignment)
        print("MLR (scikit-learn) fitted")
        print(f" - Train MSE: {self.train_mse_:.6f} | Train R^2: {self.train_r2_:.6f}")
        if self.val_split > 0.0:
            print(f" -  Val  MSE: {self.val_mse_:.6f} |  Val  R^2: {self.val_r2_:.6f}")

        return self

    def predict(self, X):
        """
        Compute predictions for new inputs.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions with shape (n_samples,) for single-output
            or (n_samples, n_targets) for multi-output.
        """
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict(X)

    def metrics(self):
        """
        Return a dictionary with stored training/validation metrics.
        Useful to include in the report directly.
        """
        return {
            "train_mse": self.train_mse_,
            "train_r2": self.train_r2_,
            "val_mse": self.val_mse_,
            "val_r2": self.val_r2_,
        }

    @property
    def coef_(self):
        """
        Access learned coefficients (after pipeline). Returns the coefficients
        of the LinearRegression step, already considering scaling if enabled.
        """
        # Get the final linear regression step
        lin = self.model.named_steps["linreg"]
        return lin.coef_

    @property
    def intercept_(self):
        """Access learned intercept."""
        lin = self.model.named_steps["linreg"]
        return lin.intercept_
