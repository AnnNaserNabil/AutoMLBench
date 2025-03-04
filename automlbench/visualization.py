### visualization.py
import matplotlib.pyplot as plt
import pandas as pd


def plot_performance(results, metrics=["Accuracy"]):
    results_df = pd.DataFrame(results).T
    results_df = results_df[metrics]
    results_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Metrics")
    plt.grid(axis="y")
    plt.show()
