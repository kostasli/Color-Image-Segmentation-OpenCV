"""
FullName: Lyeros Konstantinos
Lab: Trith 18:00-20:00
Academic Email: cse47429@uniwa.gr
AM: 71347429
"""

import cv2
import numpy as np
import random


# function to illustrate the production of random noise


def randNoise(prob, img):
    ns = np.zeros(img.shape, np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rnd1 = random.random()
            rnd2 = random.random()
            ns[i][j] = img[i][j] + prob * rnd1 * img[i][j] - prob * rnd2 * img[i][j]
    return ns

# function to illustrate template matching


def templateMatching(img, temp):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = temp.shape[::-1]
    # run the template matching
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.72
    loc = np.where(res >= threshold)
    # mark the corresponding location(s)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    return img

# read the template image


template = cv2.imread('piece.jpg', 0)

# execute template matching for given noise probabilities
for prb in [0, 0.1, 0.15, 0.2]:
    # load the template image we look for
    image = cv2.imread('pieces.jpg')
    noise = randNoise(prb, image)
    match = templateMatching(noise, template)
    cv2.namedWindow('Template matching with ' + str(prb * 100) + '% noise.', cv2.WINDOW_NORMAL)
    cv2.imshow('Template matching with ' + str(prb * 100) + '% noise.', match)
    cv2.resizeWindow('Template matching with ' + str(prb * 100) + '% noise.', 480, 360)
    cv2.waitKey(0)


# repeat with gaussian filter
for prb in [0.1, 0.15, 0.2]:
    # load the template image we look for
    image = cv2.imread('pieces.jpg')
    noise = randNoise(prb, image)
    blur = cv2.GaussianBlur(noise, (5, 5), 0)
    match = templateMatching(blur, template)
    cv2.namedWindow('Template matching with ' + str(prb * 100) + '% noise and Gaussian Blur.', cv2.WINDOW_NORMAL)
    cv2.imshow('Template matching with ' + str(prb * 100) + '% noise and Gaussian Blur.', match)
    cv2.resizeWindow('Template matching with ' + str(prb * 100) + '% noise and Gaussian Blur.', 480, 360)
    cv2.waitKey(0)


cv2.destroyAllWindows()
