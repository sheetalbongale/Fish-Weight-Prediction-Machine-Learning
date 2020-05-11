# ============ Dependencies and packages =============#
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocess import prep_data
# ------------------ sci-kit learn ------------------#
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
# ----------------- joblib dump & load --------------#
from joblib import dump
from joblib import load
#=====================================================#


# ============= Load Data $ Train Model ==============#
csv_path = os.path.join("data/fish_participant.csv")
df = pd.read_csv(csv_path)

X, y = prep_data(df)

gbr = GradientBoostingRegressor(
    n_estimators=100, max_depth=8, min_samples_split=2, learning_rate=0.1, loss="ls",
)

gbr.fit(X, y)

# ==================== Save model =====================#
dump(gbr, "reg.joblib")


# ================= Linear Regression =================#
# Linear Regression
# lr = LinearRegression()
# lr.fit(X, y)
