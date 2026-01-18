from src.load_data import load_energy_data
from src.preprocess import preprocess_energy_data
from src.train_model import train_model
from src.evaluate import evaluate_model
from src.cross_validation import run_cross_validation
from src.visualizations import show_visualizations
from src.feature_importance import plot_feature_importance
from src.model_comparison import compare_models

DATA_PATH = "data/raw/energy-consumption-2020-1.csv"

def main():
    df = load_energy_data(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_energy_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    run_cross_validation(X_train, y_train)
    show_visualizations(df)
    plot_feature_importance(model, X_train)
    compare_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
