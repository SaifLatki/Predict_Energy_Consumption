from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def compare_models(X_train, X_test, y_train, y_test):

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_r2 = r2_score(y_test, lr.predict(X_test))

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))

    print("Model Comparison (R² Score)")
    print("--------------------------")
    print(f"Linear Regression : {lr_r2:.3f}")
    print(f"Random Forest    : {rf_r2:.3f}")
