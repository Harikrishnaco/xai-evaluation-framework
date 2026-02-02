import numpy as np

class XAIEvaluator:
    """
    Evaluator for LIME-based explanations
    Computes simple, defendable metrics:
    - Quality (existence & consistency)
    - Stability (prediction sensitivity to perturbation)
    - Usability (interpretability heuristic)
    """

    def evaluate_lime_only(self, lime_explanations, original_pred, perturbed_pred):
        """
        Parameters
        ----------
        lime_explanations : list
            List of LIME explanation objects
        original_pred : np.ndarray
            Model predictions on original samples
        perturbed_pred : np.ndarray
            Model predictions on perturbed samples
        """

        # ---------- Quality ----------
        # Check whether explanations were generated successfully
        quality_score = "Valid explanations generated" if len(lime_explanations) > 0 else "No explanations"

        # ---------- Stability ----------
        # Simple stability metric: mean absolute change in predictions
        stability_score = float(
            np.mean(np.abs(original_pred - perturbed_pred))
        )

        # ---------- Usability ----------
        # Heuristic: number of features explained per instance
        avg_features = np.mean([
            len(exp.as_list()) for exp in lime_explanations
        ])

        if avg_features <= 5:
            usability = "High"
        elif avg_features <= 10:
            usability = "Medium"
        else:
            usability = "Low"

        return {
            "quality": quality_score,
            "stability": stability_score,
            "usability": usability
        }
