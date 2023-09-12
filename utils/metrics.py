import numpy as np


# Define SMAPE loss function
def SMAPE(actual, pred):
    return 100 * np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred)))
