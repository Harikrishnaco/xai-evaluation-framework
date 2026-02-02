import lime
import lime.lime_tabular
import numpy as np

def generate_explanations(model, X_train, X_test, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=["No", "Yes"],
        mode="classification"
    )

    explanations = []
    for i in range(len(X_test)):
        exp = explainer.explain_instance(
            data_row=np.array(X_test.iloc[i]),
            predict_fn=model.predict_proba
        )
        explanations.append(exp)

    return explanations
