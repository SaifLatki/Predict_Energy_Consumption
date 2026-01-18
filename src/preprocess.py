import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_energy_data(df):

    df.columns = df.columns.str.strip()

    # Remove extreme outliers
    q_low = df['Electricity Usage'].quantile(0.01)
    q_high = df['Electricity Usage'].quantile(0.99)
    df = df[(df['Electricity Usage'] >= q_low) &
            (df['Electricity Usage'] <= q_high)]

    # Drop ID columns
    df = df.drop(columns=['Site Name', 'Address'])

    # 🎯 TARGET (LOG TRANSFORM)
    y = np.log1p(df['Electricity Usage'])

    # Features
    X = df.drop(columns=['Electricity Usage'])
    X = X.fillna(0)

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=['Department', 'Electric Utility', 'Building Type'],
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
