import pandas as pd
import numpy as np


def data_split(data_arr):
    num_rows, num_cols = data_arr.shape
    pred_data = data_arr[:, :-1].copy()
    tar_data = data_arr[:, -1]
    tar_data = tar_data.reshape(len(tar_data), 1)
    return pred_data, tar_data


def feature_normalizer(arr):
    arr_min = arr.min(axis=1, keepdims=True)  # Taking advantage of numpy broadcasting
    arr_max = arr.max(axis=1, keepdims=True)  # Smaller array is "broadcast" across the larger array
    arr_norm = (arr - arr_min) / (arr_max - arr_min)
    return arr_norm


def calculate_metrics(confusion_mtx):
    tp = np.diag(confusion_mtx)
    fn = np.sum(confusion_mtx, axis=1) - tp
    fp = np.sum(confusion_mtx, axis=0) - tp
    tn = np.sum(confusion_mtx) - (tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = np.sum(np.diag(confusion_mtx)) / np.sum(confusion_mtx)
    return precision, recall, accuracy


def confusion_matrix(test_tar, predictions):
    test_tar, predictions = test_tar.flatten(), predictions.flatten()
    uni_class = np.unique(test_tar)
    matrix = np.zeros((uni_class.shape[0], uni_class.shape[0]))
    for i, actual in enumerate(uni_class):
        for j, pred in enumerate(uni_class):
            matrix[i, j] = np.sum(np.logical_and(test_tar == actual, predictions == pred))
    return matrix


def kNN_alg(train_pred, test_pred, train_tar, test_tar):
    distances = np.zeros(shape=(len(test_tar), 9))
    mydict = dict()
    for i in range(len(test_pred)):
        point, train_pred = test_pred[i].astype(float), train_pred.astype(float)
        diff_arr = train_pred - point  # Numpy broadcast applies
        dist_arr = np.sqrt(np.sum(diff_arr * diff_arr, axis=1))
        distances[i] = np.argsort(dist_arr)[:9]
    for k in range(1, 10, 2):
        guess_class = np.zeros(shape=(len(test_tar), k))
        temp = distances[:, :k].astype(int)
        for j in range(len(temp)):
            guess_class[j] = train_tar[temp[j]].flatten()
        guess_class = guess_class.astype(int)
        predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=guess_class).reshape(
            len(test_tar), 1)
        conf_mtx = confusion_matrix(test_tar.flatten(), predictions.flatten())
        precisions, recalls, accuracy = calculate_metrics(conf_mtx)
        recall_macro_avg = np.sum(recalls) / len(recalls)
        prec_macro_avg = np.sum(precisions) / len(precisions)
        mydict[k] = {"accuracy": "{:.2%}".format(accuracy), "Recall Macro Average": "{:.2%}".format(recall_macro_avg),
                     "Precision Macro Average": "{:.2%}".format(prec_macro_avg)}
    print(mydict)


df = pd.read_csv("16P.csv", encoding="Latin1")
df = df.drop("Response Id", axis=1)

df.loc[df["Personality"] == "ESTJ", "Personality"] = 0
df.loc[df["Personality"] == "ENTJ", "Personality"] = 1
df.loc[df["Personality"] == "ESFJ", "Personality"] = 2
df.loc[df["Personality"] == "ENFJ", "Personality"] = 3
df.loc[df["Personality"] == "ISTJ", "Personality"] = 4
df.loc[df["Personality"] == "ISFJ", "Personality"] = 5
df.loc[df["Personality"] == "INTJ", "Personality"] = 6
df.loc[df["Personality"] == "INFJ", "Personality"] = 7
df.loc[df["Personality"] == "ESTP", "Personality"] = 8
df.loc[df["Personality"] == "ESFP", "Personality"] = 9
df.loc[df["Personality"] == "ENTP", "Personality"] = 10
df.loc[df["Personality"] == "ENFP", "Personality"] = 11
df.loc[df["Personality"] == "ISTP", "Personality"] = 12
df.loc[df["Personality"] == "ISFP", "Personality"] = 13
df.loc[df["Personality"] == "INTP", "Personality"] = 14
df.loc[df["Personality"] == "INFP", "Personality"] = 15

data_arr = np.array(df)
pred_data, tar_data = data_split(data_arr)
pred1, pred2, pred3, pred4, pred5 = np.array_split(pred_data, 5)
tar1, tar2, tar3, tar4, tar5 = np.array_split(tar_data, 5)

kNN_alg(np.concatenate((pred2, pred3, pred4, pred5), axis=0), pred1,
        np.concatenate((tar2, tar3, tar4, tar5), axis=0), tar1)
