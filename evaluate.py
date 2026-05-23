import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def metrics_sklearn(y_true, y_pred):
    """The evaluation measures for the regression task were calculated, including Pearson correlation coefficient (R) and mean square error (MSE)."""
    y_true=np.array(y_true).ravel()
    y_pred=np.array(y_pred).ravel()
    mse = mean_squared_error(y_true, y_pred)
    r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    return r, mse
