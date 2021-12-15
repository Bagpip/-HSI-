import numpy as np
import cv2 as cv
from sklearn import model_selection
from sklearn.svm import SVC
import os
import utilities_test
import warnings
from thundersvm import SVC
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path_blank_hdr = 'G:/date/L/L/hyper/031368c-20x-blank.hdr'
path_blank_raw = 'G:/date/L/L/hyper/031368c-20x-blank.raw'
path_hdr = 'G:/date/L/L/hyper/031368c-20x-roi2.hdr'
path_raw = 'G:/date/L/L/hyper/031368c-20x-roi2.raw'
label_path = "G:/date/L/L/annotation/031368c-20x-roi2.xml"
dim = 60
size = 0.01

X = []
Y = []
Tr, Te, Z = utilities_test.split(path_hdr, path_raw, label_path, path_blank_hdr, path_blank_raw, size, dim)

X = Z[:, 0:dim]
Y = Z[:, dim]
Tr = utilities_test.date_std(Tr)
X = utilities_test.date_std(X)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=5)

# model = SVC(C=100.0, kernel='poly', degree=2, gamma=0.017, coef0=0.0, shrinking=True, probability=False, tol=0.001,
#             cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
model = SVC(kernel='polynomial', degree=2, gamma=0.017, coef0=0.0, C=10.0, tol=0.001, probability=False,
            class_weight=None,
            shrinking=True, cache_size=200, verbose=False, max_iter=-1, n_jobs=-1, max_mem_size=-1,
            random_state=None,
            decision_function_shape='ovo', gpu_id=0)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
model.fit(X_train, Y_train)
y_test_pred = model.predict(Tr)

print("#" * 30)
print("\nClassification report on test dataset\n")
print(classification_report(Te, y_test_pred))
print("#" * 30 + "\n")


y_test_pred = np.reshape(y_test_pred, (-1, 1))

pre_img = np.reshape(y_test_pred, (1024, 1280))
for i in range(pre_img.shape[0]):
    for j in range(pre_img.shape[1]):
        if pre_img[i, j] == 1:
            pre_img[i, j] = 255
        else:
            pre_img[i, j] = 0
pre_img = np.array(pre_img, dtype=np.uint8)

cv.namedWindow("demo00", cv.WINDOW_NORMAL)
cv.imshow("demo00", pre_img)

k = np.ones((10, 10), np.uint8)

r=cv.bilateralFilter(pre_img, 25, 100, 100)

cv.namedWindow("demo02", cv.WINDOW_NORMAL)
cv.imshow("demo02", r)

pre_img1 = cv.morphologyEx(pre_img, cv.MORPH_CLOSE, k, iterations=3)

cv.namedWindow("demo02", cv.WINDOW_NORMAL)
cv.imshow("demo02", pre_img1)



print("#" * 30)
print("\nClassification report on test dataset\n")
print(classification_report(Y, y_test_pred))
print("#" * 30 + "\n")

''''
view = imshow(img_low, (29, 19, 9))
label1 = label_test.draw_label(label_path)
cv.namedWindow("demo1", cv.WINDOW_NORMAL)
cv.imshow("demo1", label1)

view1 = imshow(img_high, (29, 19, 9))
label2 = label_test.draw_label(label_path1)
cv.namedWindow("demo2", cv.WINDOW_NORMAL)
cv.imshow("demo2", label2)

img = []
for i in range(img1.shape[2]):
    img_raw_part = img1[:, :, i]
    img_part_blank = img1[:, :, i]
    img.append(img_raw_part)

img = np.array(img)
img_max = img.max(axis=0)
img_max_max = img_max.max(axis=0)
print(img_max_max.max(axis=0))
view = imshow(img)
# img_gt = envi.open('E:/brain_cancer/HSI_Human_Brain_Database_IEEE_Access/004-02/gtMap.hdr',
#                      'E:/brain_cancer/HSI_Human_Brain_Database_IEEE_Access/004-02/gtMap')
Y = label_test.getlabel(label_path, 1, 0.1)

print(img)

# print(img_gt)

img_out = np.zeros(shape=(389, 345, 826))

gt = open_image(path_hdr).read_band(0)
shrink = []
for i in range(60):
    img_shrink = img[:, :, i]
    size = (int(img.shape[0] * 0.3), int(img.shape[1] * 0.3))
    shrink_1 = cv.resize(img_shrink, size)
    shrink.append(shrink_1)
view = imshow(classes=gt)
view = imshow(shrink, (30, 20, 10), classes=gt)
view.set_display_mode('overlay')
view.class_alpha = 0.5

img_out_reshape = np.reshape(img_out, (-1, 826))
pca = PCA(n_components=129)  # 将826维度降到129维度
img_out_reshape_pca = pca.fit_transform(img_out_reshape)
img_out_reshape_pca_final = np.reshape(img_out_reshape_pca, (389, 345, 129))
img_inverse = pca.inverse_transform(img_out_reshape_pca)  # 将降维后的129维度重新映射到826维，查看住份重构图
img_inverse_reshape = np.reshape(img_inverse, (389, 345, 826))

w = view_nd(img_inverse_reshape[:, :, 15], classes=gt)

#  需要安装wx包，openGL包
# pip install wxPython
# pip install PyOpenGL
view_cube(img, bands=[])  # 超立方体
view_cube(img, bands=[29, 19, 9])  # 指定波段运算更快
# img_1 = img[:, :, 229].reshape(389, 345)
# view = imshow(img, (29, 19, 9))
cv.waitKey(0)
'''