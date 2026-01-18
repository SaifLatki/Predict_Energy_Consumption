import matplotlib.pyplot as plt
import seaborn as sns

def show_visualizations(df):

    # 1. Electricity usage distribution
    plt.figure()
    sns.histplot(df['Electricity Usage'], bins=30)
    plt.title("Electricity Usage Distribution")
    plt.xlabel("Electricity Usage")
    plt.ylabel("Count")
    plt.show()

    # 2. Building Area vs Electricity Usage
    plt.figure()
    sns.scatterplot(
        x=df['Building Area'],
        y=df['Electricity Usage']
    )
    plt.title("Building Area vs Electricity Usage")
    plt.xlabel("Building Area")
    plt.ylabel("Electricity Usage")
    plt.show()
