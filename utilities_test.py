import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import cv2 as cv
import label_test


def date_std(X):
    standardscaler = StandardScaler()
    standardscaler.fit(X)
    X = standardscaler.transform(X)
    return X


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def normalize_img(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))
    image = (image - mean) / np.sqrt(var)
    return image


def split(path_hdr, path_raw, path_label, path_hdr_blank, path_raw_blank, size=0.01, dim=16):
    img = envi.open(path_hdr, path_raw)
    img_blank = envi.open(path_hdr_blank, path_raw_blank)

    channel = img.shape[2]

    img_raw = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    img_raw[:, :, :] = img[:, :, :]
    img_raw_blank = np.zeros((img_blank.shape[0], img_blank.shape[1], img_blank.shape[2]))
    img_raw_blank[:, :, :] = img_blank[:, :, :]

    img1 = label_test.getlabel(path_label)
    img_label = np.zeros((img1.shape[0], img1.shape[1]))
    img_label[:, :] = img1[:, :]

    # 多维训练集
    X = np.zeros((img_raw.shape[0] * img_raw.shape[1], img_raw.shape[2]))
    for i in range(channel):
        img_reshape = img_raw[:, :, i] / img_raw_blank[:, :, i]
        X_reshape = np.reshape(img_reshape, (-1, 1))
        X[:, i] = X_reshape[:, -1]

    # 一维标签
    Y = np.reshape(img_label, (-1, 1))

    # 降维
    pca = PCA(n_components=dim)
    X = pca.fit_transform(X)

    Z = np.zeros(shape=(X.shape[0], X.shape[1] + 1))
    Z[:, 0:X.shape[1]] = X[:, 0:X.shape[1]]
    Z[:, X.shape[1]] = Y[:, -1]

    # 取样
    size = int(size * Z.shape[0])
    F = np.zeros(shape=(size, Z.shape[1]))
    array = np.array([0, 0])
    rand_arr = np.arange(Z.shape[0])
    np.random.shuffle(rand_arr)
    F = Z[rand_arr[0:size]]

    return X, Y, F
