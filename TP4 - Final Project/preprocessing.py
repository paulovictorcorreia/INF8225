# %%
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split

# %% Importing data

RANDOM_SEED = 42
TEST_SIZE = 0.3
VALIDATION_SIZE = 0.5

eeg_signal = pd.read_csv("data/eeg_signals.csv", index_col=0)
eeg_signal["y"] = (eeg_signal["y"] == 1).astype(np.int32)

# %% Splitting data in training, validation and testing

X_train, X_test, y_train, y_test = train_test_split(
    eeg_signal.drop("y", axis=1), eeg_signal["y"], test_size=TEST_SIZE,
    stratify=eeg_signal["y"], random_state=42
)

X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=0.5,
    stratify=y_test, random_state=RANDOM_SEED
)

# %% Formatting data to time series formating
X_train = to_time_series_dataset(X_train)
X_test = to_time_series_dataset(X_test)
X_val = to_time_series_dataset(X_val)

# %% normalizing time-series
scaler = TimeSeriesScalerMeanVariance()
# scaler = TimeSeriesScalerMinMax()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
