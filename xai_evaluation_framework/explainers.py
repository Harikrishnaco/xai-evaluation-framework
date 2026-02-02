from omnixai.explainers.tabular import TabularExplainer

def generate_explanations(model, X_train, X_test, feature_names):
    explainer = TabularExplainer(
        explainers=["shap", "lime"],
        model=model,
        data=X_train,
        feature_names=feature_names,
        mode="classification"
    )

    explanations = explainer.explain(X_test[:10])
    return explanations
