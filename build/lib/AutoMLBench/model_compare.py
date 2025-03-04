def compare_models(results):
    """Ranks models based on accuracy."""
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(sorted_results):
        print(f"{i+1}. {model}: {score:.4f}")
