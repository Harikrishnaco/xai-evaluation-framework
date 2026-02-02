def fidelity_score(original_pred, perturbed_pred):
    return abs(original_pred - perturbed_pred)

def feature_agreement(shap_features, lime_features, k=5):
    return len(set(shap_features[:k]) & set(lime_features[:k])) / k
