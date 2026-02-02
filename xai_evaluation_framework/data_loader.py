import pandas as pd

def load_pima():
    df = pd.read_csv("datasets/pima_diabetes.csv")
    return df

def load_student():
    df = pd.read_csv("datasets/student_performance.csv")
    return df
