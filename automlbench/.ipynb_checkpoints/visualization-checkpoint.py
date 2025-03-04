import matplotlib.pyplot as plt

def plot_results(results, metric='Accuracy'):
    """Plots model performance for a given metric."""
    models = list(results.keys())
    scores = [results[model].get(metric, 0) for model in models]

    plt.figure(figsize=(10, 6))
    plt.barh(models, scores, color="skyblue")
    plt.xlabel(metric)
    plt.ylabel("Models")
    plt.title(f"Model Performance Comparison ({metric})")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()
