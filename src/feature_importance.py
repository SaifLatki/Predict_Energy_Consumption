import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, X_train):

    importance = model.feature_importances_

    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).head(10)

    plt.figure()
    plt.barh(features_df['Feature'], features_df['Importance'])
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.show()
