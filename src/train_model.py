from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # ✅ Save model AND feature names
    joblib.dump(
        {"model": model, "features": X_train.columns.tolist()},
        "energy_model.pkl"
    )

    return model
