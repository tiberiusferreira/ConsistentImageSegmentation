import numpy as np
from sklearn.decomposition import PCA
import cv2


def apply_pca(img):
    data = list()
    row, col = np.shape(img)
    img = img[15:row - 15, 15:col - 15]
    row, col = np.shape(img)
    for x in range(row):
        for y in range(col):
            if img[x][y] > 0:
                data.append([x - (row / 2), y - (col / 2)])
    X = data
    pca = PCA(n_components=2)
    pca.fit(X)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print ('Drawing line')
    pca_comp = pca.components_ * 60
    print(pca_comp)
    offset = 50
    pca_comp += offset
    cv2.line(img, (offset, offset),
             (int(pca_comp[0][0]), int(pca_comp[0][1])), (255, 0, 0), 5)
    cv2.line(img, (offset, offset),
             (int(pca_comp[1][0]), int(pca_comp[1][1])), (0, 255, 0), 5)
    # cv2.line(img, (1, 1), (100, 10), (255, 0, 0), 5)
    cv2.imshow('RGB_img', img)
    cv2.waitKey(1)


def apply_sobel(objs):
    if objs is not None:
        for tuples in objs:
            rect, center = tuples
            cv2.imshow('Rct', rect)
            cv2.waitKey(1)
            img_bgr8_clean_copy = rect.copy()
            Sobelx = cv2.Sobel(img_bgr8_clean_copy, -1, 1, 0, ksize=1)
            Sobely = cv2.Sobel(img_bgr8_clean_copy, -1, 0, 1, ksize=1)
            Sobel = Sobelx + Sobely
            img_bgr8_clean_copy = Sobel
            kernel = np.ones((5, 5), np.uint8)
            img_bgr8_clean_copy = cv2.cvtColor(img_bgr8_clean_copy, cv2.COLOR_BGR2GRAY)
            img_bgr8_clean_copy = cv2.morphologyEx(img_bgr8_clean_copy, cv2.MORPH_GRADIENT, kernel)

            # img_bgr8_clean_copy = cv2.GaussianBlur(img_bgr8_clean_copy, (7, 7), 0)



            ret3, img_bgr8_clean_copy = cv2.threshold(img_bgr8_clean_copy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow('Result', img_bgr8_clean_copy)
            # cv2.waitKey(1)
            apply_pca(img_bgr8_clean_copy)

