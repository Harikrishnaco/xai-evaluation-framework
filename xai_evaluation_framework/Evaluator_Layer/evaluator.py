from .quality_metrics import fidelity_score, feature_agreement
from .stability_metrics import stability_score
from .usability_metrics import usability_score

class XAIEvaluator:

    def evaluate_quality(self, shap_exp, lime_exp, original_pred, perturbed_pred):
        quality = {}
        quality["fidelity"] = fidelity_score(
            original_pred, perturbed_pred
        )
        quality["feature_agreement"] = feature_agreement(
            shap_exp["features"], lime_exp["features"]
        )
        return quality

    def evaluate_stability(self, shap_orig, shap_pert):
        return {
            "stability_score": stability_score(shap_orig, shap_pert)
        }

    def evaluate_usability(self, num_features):
        return {
            "usability": usability_score(num_features)
        }

    def evaluate_all(
        self,
        shap_exp,
        lime_exp,
        shap_orig,
        shap_pert,
        original_pred,
        perturbed_pred
    ):
        return {
            "quality": self.evaluate_quality(
                shap_exp, lime_exp, original_pred, perturbed_pred
            ),
            "stability": self.evaluate_stability(
                shap_orig, shap_pert
            ),
            "usability": self.evaluate_usability(
                len(shap_exp["features"])
            )
        }
