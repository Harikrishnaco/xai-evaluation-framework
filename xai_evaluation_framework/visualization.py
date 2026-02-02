import matplotlib.pyplot as plt

def plot_stability(stability_scores):
    methods = list(stability_scores.keys())
    scores = list(stability_scores.values())

    plt.figure()
    plt.bar(methods, scores)
    plt.xlabel("XAI Method")
    plt.ylabel("Stability Score (Lower is Better)")
    plt.title("Stability Comparison of XAI Methods")
    plt.show()
