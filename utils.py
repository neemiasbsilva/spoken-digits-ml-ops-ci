import tensorflow as tf
import numpy as np
import hub
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def save_json(path, data):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

        print("Json sucessfully saved!")
    except Exception as e:
        print(e)


def load_dataset(data_path):
    # ds = hub.load("hub://activeloop/spoken_mnist")
    ds = hub.load(data_path)
    data, target = ds.spectrograms[:], ds.labels[:]

    return data, target


def split_train_test(data, target, test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=7)

    return (X_train, y_train), (X_test, y_test)


def save_metrics(json_path, y_true, y_pred):
    acc = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    return 

# data, target = load_dataset()

# data = np.asarray(data)
# target = np.asarray(target)

# train_dataset, test_dataset = split_train_test(data, target)

# X_train, y_train = train_dataset
# X_test, y_test = test_dataset
# print(data.shape, target.shape)
# print(X_train.shape, X_test.shape)
