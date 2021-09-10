"""
FullName: Lyeros Konstantinos
Lab: Trith 18:00-20:00
Academic Email: cse47429@uniwa.gr
AM: 71347429
"""

import cv2
import numpy as np


# color histogram similarity function
def colorHistSimilarity(imgPatch, imgTemp, distanceFunction):
    templateHist = cv2.calcHist([imgTemp], [0], None, [256], [0, 256])
    imgPatchHist = cv2.calcHist([imgPatch], [0], None, [256], [0, 256])
    histScore = cv2.compareHist(templateHist, imgPatchHist, distanceFunction)
    return histScore


# function to illustrate template matching
def templateMatching(img, temp, distFunc, thres):
    # dimensions of images
    h, w = img.shape[:2]
    m, n = temp.shape[:2]
    matchingValues = np.zeros((h - m, w - n))
    loc = []
    mask = np.ones((m, n, 3))
    maxScores = []
    # template-window creation
    for i in range(0, h - m):
        for j in range(w - n):
            imagePatch = image[i:i + m, j:j + n]
            matchingValues[i, j] = colorHistSimilarity(imagePatch, temp, distFunc)
            if matchingValues[i][j] > thres:
                maxScores.append(matchingValues[i, j])
                offset = np.array((i, j))
                img[offset[0]:offset[0] + mask.shape[0], offset[1]:offset[1] + mask.shape[1]] = mask
                (max_Y, max_X) = (i, j)
                loc.append((max_X, max_Y))
    return loc, maxScores

# read the images and store the dimensions of template


image = cv2.imread('presents.jpg')
image2 = cv2.imread('presents.jpg')
image3 = cv2.imread('presents.jpg')
template = cv2.imread('present.jpg')
height, width = template.shape[:2]

# get user input for threshold
threshold = input('Choose a correlation threshold in range of 0 to 1.')
threshold = float(threshold)

# correlation distance function
topLeft, scores = templateMatching(image, template, 0, threshold)
for i in range(0, len(topLeft)):
    loc = topLeft[i]
    cv2.rectangle(image2, loc, (loc[0] + width, loc[1] + height), (0, 0, 255), 3)

cv2.namedWindow('Template matching using Correlation function.', cv2.WINDOW_NORMAL)
cv2.imshow('Template matching using Correlation function.', image2)
cv2.resizeWindow('Template matching using Correlation function.', 480, 360)
cv2.waitKey(0)
cv2.destroyWindow('Template matching using Correlation function.')

# read the original image again

image = cv2.imread('presents.jpg')

# get user input for threshold
threshold = input('Choose an intersection threshold in range of 9000 to 10000.')
threshold = float(threshold)

# intersection function
topLeft, scores = templateMatching(image, template, 2, threshold)
for i in range(0, len(topLeft)):
    loc = topLeft[i]
    cv2.rectangle(image3, loc, (loc[0] + width, loc[1] + height), (0, 0, 255), 3)

cv2.namedWindow('Template matching using Intersection function.', cv2.WINDOW_NORMAL)
cv2.imshow('Template matching using Intersection function.', image3)
cv2.resizeWindow('Template matching using Intersection function.', 480, 360)
cv2.waitKey(0)

cv2.destroyAllWindows()
