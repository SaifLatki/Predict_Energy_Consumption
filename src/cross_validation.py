from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def run_cross_validation(X, y):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring='r2'
    )

    print("Cross-Validation R2 Scores:", scores)
    print("Mean R2:", np.mean(scores))
