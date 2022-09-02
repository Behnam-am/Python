import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split


def fetch_kaggle():
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_file("uciml/pima-indians-diabetes-database",
                                  file_name="diabetes.csv",
                                  path="datasets/diabetes")
    except:
        print("unable to load fetch kaggle")

def load_data():
    path = os.path.join("datasets", "diabetes")
    csv_path = path + "/diabetes.csv"
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    fetch_kaggle()
    data = load_data()
    data.hist(bins=10, figsize=(15, 10))
    plt.show()

    train_set, test_set = train_test_split(data, test_size=0.2, random_state=0,
                                           stratify=data["Outcome"])



