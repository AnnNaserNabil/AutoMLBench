import matplotlib.pyplot as plt
from visualization import plot_results  # Importing from visualization.py

def compare_models(results, metric='Accuracy'):
    """Ranks models based on a selected metric."""
    sorted_results = sorted(results.items(), key=lambda x: x[1].get(metric, 0), reverse=True)

    print(f"\nModel Ranking based on {metric}:\n")
    for i, (model, scores) in enumerate(sorted_results, start=1):
        print(f"{i}. {model}: {scores.get(metric, 0):.4f}")

    # Optionally, visualize the results
    plot_results(results, metric)
