#===================== Packages =====================#
from joblib import load
from preprocess import prep_data
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

#================== Predict Function ==================#
def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    X, y = prep_data(df)

    reg = load("reg.joblib")

    predictions = reg.predict(X)

    return predictions

#========================= Main ========================#
if __name__ == "__main__":
    predictions = predict_from_csv("data/fish_holdout_demo.csv")
    y_truth = pd.read_csv("data/fish_holdout_demo.csv")["Weight"].values

    ho_mse = mean_squared_error(y_truth, predictions)
    r2_score = r2_score(y_truth, predictions)

    print(y_truth)
    print("PREDICTIONS:")
    print(predictions)
    print("_______ Score Metrics _______")
    print("MSE:", ho_mse)
    print("r2_score:", r2_score)
    print("_____________________________")


#================== Test fish_holdout.csv ==================#
#    from sklearn.metrics import mean_squared_error
#    ho_predictions = predict_from_csv("fish_holdout.csv")
#    ho_truth = pd.read_csv("fish_holdout.csv")["Weight"].values
#    ho_mse = mean_squared_error(ho_truth, ho_predictions)
#    print(ho_mse)
