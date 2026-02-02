import os
import pandas as pd

# Base directory of this file
BASE_DIR = os.path.dirname(__file__)

def load_pima():
    """
    Load PIMA Indians Diabetes dataset
    """
    path = os.path.join(BASE_DIR, "datasets", "pima_diabetes_data.csv")
    df = pd.read_csv(path)
    return df


def load_student():
    """
    Load Student Performance dataset
    """
    path = os.path.join(BASE_DIR, "datasets", "student_data.csv")
    df = pd.read_csv(path)
    return df
