import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

def preprocess_pima(df):
    # Replace zero values with median (domain knowledge)
    cols_with_zero = ['Glucose', 'BloodPressure', 'BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, df[col].median())
    return df

def preprocess_student(df):
    # Convert final grade to Pass/Fail
    df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G1', 'G2', 'G3'])
    return df
