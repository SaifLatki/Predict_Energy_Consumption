import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):

    preds_log = model.predict(X_test)

    # 🔄 Reverse log
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test)

    print("Model Performance (Improved)")
    print("----------------------------")
    print("MAE :", mean_absolute_error(y_true, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, preds)))
    print("R2  :", r2_score(y_true, preds))
