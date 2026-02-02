# run_experiments.py

# -------- Imports (relative, required for -m execution) --------
from .data_loader import load_pima
from .data_preprocessing import preprocess_pima, preprocess_data
from .model_training import train_random_forest
from .explainers import generate_explanations

from .evaluation.evaluator import XAIEvaluator
from .evaluation.stability_metrics import add_noise

# -------- 1. Load & preprocess dataset --------
df = load_pima()
df = preprocess_pima(df)

X_train, X_test, y_train, y_test = preprocess_data(df, target="Outcome")

# -------- 2. Train model --------
model, acc = train_random_forest(X_train, y_train, X_test, y_test)
print("Model Accuracy:", acc)

# -------- 3. Generate explanations (LIME only) --------
feature_names = df.drop(columns=["Outcome"]).columns.tolist()

X_sample = X_test[:5]

lime_explanations = generate_explanations(
    model, X_train, X_sample, feature_names
)

# -------- 4. Predictions (for fidelity / stability proxy) --------
original_pred = model.predict_proba(X_sample)[:, 1]

X_perturbed = add_noise(X_sample, noise_level=0.01)
perturbed_pred = model.predict_proba(X_perturbed)[:, 1]

# -------- 5. Evaluate explanations --------
evaluator = XAIEvaluator()

results = evaluator.evaluate_lime_only(
    lime_explanations=lime_explanations,
    original_pred=original_pred,
    perturbed_pred=perturbed_pred
)

# -------- 6. Print results --------
print("\nModel Accuracy:", acc)
print("\n===== XAI Evaluation Results =====")
print("Quality Metrics :", results["quality"])
print("Stability Metrics :", results["stability"])
print("Usability Metrics :", results["usability"])

