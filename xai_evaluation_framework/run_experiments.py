from evaluation.evaluator import XAIEvaluator
from evaluation.stability_metrics import add_noise

evaluator = XAIEvaluator()

# Take a small sample for stability test
X_sample = X_test[:5]

# Original prediction
original_pred = model.predict_proba(X_sample)[:, 1]

# Original explanations
original_explanations = generate_explanations(
    model, X_train, X_sample, feature_names
)

# Perturb input
X_perturbed = add_noise(X_sample, noise_level=0.01)

# Prediction after perturbation
perturbed_pred = model.predict_proba(X_perturbed)[:, 1]

# Perturbed explanations
perturbed_explanations = generate_explanations(
    model, X_train, X_perturbed, feature_names
)

results = evaluator.evaluate_all(
    shap_exp=original_explanations["shap"],
    lime_exp=original_explanations["lime"],
    shap_orig=original_explanations["shap"].values,
    shap_pert=perturbed_explanations["shap"].values,
    original_pred=original_pred,
    perturbed_pred=perturbed_pred
)

print("\nXAI Evaluation Results")
print("----------------------")
print("Quality Metrics:", results["quality"])
print("Stability Metrics:", results["stability"])
print("Usability Metrics:", results["usability"])

from visualization import plot_stability

plot_stability({
    "SHAP": results["stability"]["stability_score"],
    "LIME": results["stability"]["stability_score"]  # later separate
})
