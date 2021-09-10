"""
FullName: Lyeros Konstantinos
Lab: Trith 18:00-20:00
Academic Email: cse47429@uniwa.gr
AM: 71347429
"""


import cv2
import numpy as np
import skimage.measure
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans
from sklearn.metrics import accuracy_score
import random


# function to illustrate the production of random noise
def addRandomNoise(prb, img):
    ns = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd1, rnd2 = random.random(), random.random()
            ns[i][j] = img[i][j] + prb * rnd1 * img[i][j] - prb * rnd2 * img[i][j]
    return ns


# function for meanshift clustering on image
def meanShiftImage(img):
    imgShape = img.shape

    # convert image into array
    flatImg = np.reshape(img, [-1, 3])

    # calculate bandwidth for meanshift
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
    mes = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # performing meanshift on flatImg
    print('Using MeanShift algorithm.')
    mes.fit(flatImg)
    labels = mes.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("Number of estimated clusters : %d" % n_clusters_)

    # display segmented image
    segmentedImg = np.reshape(labels, imgShape[:2])

    for i in range(segmentedImg.shape[0]):
        for j in range(segmentedImg.shape[1]):
            if segmentedImg[i][j] != 0:
                segmentedImg[i][j] = 255

    return segmentedImg.astype(np.uint8)


# function for kmeans clustering on image
def kMeansImage(img, nClusters):

    print('Using Kmeans algorithm.')
    flatImg = np.reshape(img, [-1, 3])
    imgShape = img.shape
    km = MiniBatchKMeans(n_clusters=nClusters)
    km.fit(flatImg)
    labels = km.labels_

    # display segmented image
    segmentedImg = np.reshape(labels, imgShape[:2])

    for i in range(segmentedImg.shape[0]):
        for j in range(segmentedImg.shape[1]):
            if segmentedImg[i][j] != 0:
                segmentedImg[i][j] = 255

    return segmentedImg.astype(np.uint8)


# read original image
imageRGB = cv2.imread('aircraft.jpg')

# read binary annotated image
imageBinary = cv2.imread('aircraftBinary.jpg', 0)
flatBinaryImg = imageBinary.flatten()

for prob in [0, 0.05, 0.1, 0.15, 0.2]:
    noise = addRandomNoise(prob, imageRGB)
    ms = meanShiftImage(noise)
    msFlatten = ms.flatten()
    km = kMeansImage(noise, 2)
    kmFlatten = km.flatten()

    cv2.namedWindow("MeanShift clustering algorithm with noise: {:.4f}".format(prob*100) + "%", cv2.WINDOW_NORMAL)
    cv2.imshow("MeanShift clustering algorithm with noise: {:.4f}".format(prob*100) + "%", ms)
    cv2.resizeWindow("MeanShift clustering algorithm with noise: {:.4f}".format(prob*100) + "%", 480, 360)
    # cv2.imwrite('ms.jpg', ms)
    cv2.waitKey(0)
    cv2.destroyWindow("MeanShift clustering algorithm with noise: {:.4f}".format(prob*100) + "%")

    cv2.namedWindow("KMeans clustering algorithm with noise: {:.4f}".format(prob*100) + "%", cv2.WINDOW_NORMAL)
    cv2.imshow("KMeans clustering algorithm with noise: {:.4f}".format(prob*100) + "%", km)
    cv2.resizeWindow("KMeans clustering algorithm with noise: {:.4f}".format(prob*100) + "%", 480, 360)
    # cv2.imwrite('km.jpg', km)
    cv2.waitKey(0)
    cv2.destroyWindow("KMeans clustering algorithm with noise: {:.4f}".format(prob*100) + "%")

    # evaluate clustering using a ssim score metric
    accMs = accuracy_score(flatBinaryImg, msFlatten)
    accKm = accuracy_score(flatBinaryImg, kmFlatten)
    print("Accuracy score for MeanShift: {:.2f}".format(accMs*100)+"%")
    print("Accuracy score for KMeans: {:.2f}".format(accKm*100)+"%")

cv2.destroyAllWindows()
