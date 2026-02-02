import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "datasets", "pima_diabetes_data.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------------------------------------------------
# 2. Train model
# -------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# -------------------------------------------------
# 3. Wrap data for OmniXAI
# -------------------------------------------------
tabular_data = Tabular(
    X,
    feature_columns=X.columns
)

# -------------------------------------------------
# 4. Create OmniXAI explainer (DO NOT call explain())
# -------------------------------------------------
explainer = TabularExplainer(
    explainers=["lime"],
    model=model,
    data=tabular_data,
    mode="classification"
)

print("\n✅ OmniXAI explainer successfully created")
print("Explainers available:", explainer.explainers.keys())

# -------------------------------------------------
# 5. Manually generate LIME explanation via OmniXAI backend
# -------------------------------------------------
lime_backend = explainer.explainers["lime"].explainer

sample = X.iloc[0].values

lime_exp = lime_backend.explain_instance(
    data_row=sample,
    predict_fn=model.predict_proba
)

print("\n===== OmniXAI LIME Explanation Output =====")
for feature, weight in lime_exp.as_list():
    print(f"{feature}: {weight:.4f}")
