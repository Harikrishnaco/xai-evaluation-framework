
import numpy as np

def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def explanation_distance(exp1, exp2):
    """
    Euclidean distance between explanation vectors
    """
    return np.linalg.norm(exp1 - exp2)

def stability_score(original_exp, perturbed_exp):
    """
    Lower score = more stable explanation
    """
    return explanation_distance(original_exp, perturbed_exp)
