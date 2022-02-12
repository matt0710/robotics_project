import cv2
import numpy as np

img1 = cv2.imread("ImagesTrain/watch_dogs.jpg", 0)  # NB: we're importing images in greyscale
img2 = cv2.imread("ImagesQuery/watch_dogs_gt.jpg", 0)

height, width = img2.shape
print(width)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []


objp = np.zeros((1, height*width, 3), np.float32)
objp[0, :, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

# ORB --> THE DETECTOR WHOSE ROLE IS FINDING FEATURES (we have to think as feature distinctive parts of images -->
# out aim is to build a DESCRIPTOR, which is the one that convert features in computer language

orb = cv2.ORB_create(nfeatures=1000)  # feature which it has to extract

kp1, des1 = orb.detectAndCompute(img1, None)  # keypoint, descriptors
kp2, des2 = orb.detectAndCompute(img2, None)  # keypoint, descriptors

imgKp1 = cv2.drawKeypoints(img1, kp1, None)  # we draw keypoints
imgKp2 = cv2.drawKeypoints(img2, kp2, None)  # we draw keypoints

print(des1)
print(des1.shape)  # (500,32) 500 are the feature that ORB tries to extract and, for each feature, it will describe it
# using 32 values.

# a good solution is matching features to extract similarity of descriptor --> instead of using BRUTE_FORCE matcher,
# which iterates through all the row of tha first descriptor and matches it with all rows of the second one,
# so O(n^2) complexity, we will use another one which uses k-neighbor approach

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2) # k=2 implies using m, n

good = []

for m, n in matches:
    if m.distance < 0.75*n.distance:  # only in this condition we consider the matching good
        good.append([m])
        #imgpoints.append([des2[mat.trainIdx].pt for mat in matches])
        imgpoints.append(des2[m.queryIdx])
        #print([mat.trainIdx for mat in matches])
        #print(n.distance)
        objpoints.append(objp)

print(len(objpoints))
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow("kp1", imgKp1)
cv2.imshow("kp2", imgKp2)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("img3", img3)

cv2.waitKey(0)


#______________________________________________________________________________________________________________________



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img2.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)