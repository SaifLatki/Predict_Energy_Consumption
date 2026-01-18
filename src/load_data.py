import pandas as pd

def load_energy_data(file_path="/data/raw/energy-consumption-2020-1.csv"):
    
    df=pd.read_csv(file_path)

    return df
