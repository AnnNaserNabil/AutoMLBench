import matplotlib.pyplot as plt

def plot_results(results):
    """Plots model performance."""
    models, scores = zip(*results.items())
    plt.barh(models, scores, color="skyblue")
    plt.xlabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.show()
