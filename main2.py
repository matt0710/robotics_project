import cv2
import numpy as np

img = cv2.imread("ImagesTrain/ac.jpg", 0)  # NB: we're importing images in greyscale
cap = cv2.VideoCapture(0)

# ORB --> THE DETECTOR WHOSE ROLE IS FINDING FEATURES (we have to think as feature distinctive parts of images -->
# out aim is to build a DESCRIPTOR, which is the one that convert features in computer language

orb = cv2.ORB_create(nfeatures=1000)  # feature which it has to extract

kp1, des1 = orb.detectAndCompute(img, None)  # keypoint, descriptors

imgKp1 = cv2.drawKeypoints(img, kp1, None)  # we draw keypoints

bf = cv2.BFMatcher()
# print(des1)
# print(des1.shape)  # (500,32) 500 are the feature that ORB tries to extract and, for each feature, it will describe it
# using 32 values.

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(gray_frame, None)

    imgKp2 = cv2.drawKeypoints(gray_frame, kp2, None)  # we draw keypoints

    cv2.imshow("kp2", imgKp2)
    cv2.imshow("kp1", imgKp1)

    # a good solution is matching features to extract similarity of descriptor --> instead of using BRUTE_FORCE matcher,
    # which iterates through all the row of tha first descriptor and matches it with all rows of the second one,
    # so O(n^2) complexity, we will use another one which uses k-neighbor approach

    matches = bf.knnMatch(des1, des2, k=2)  # k=2 implies using m, n

    good = []
    minVal = 1000000
    kpCord = (123456789, 123456789)
    kpList = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # only in this condition we consider the matching good
            good.append([m])
            if m.distance < minVal:
                minVal = m.distance
                kpCord = kp2[n.trainIdx].pt
                kpList.append(kpCord)
        # else:
        #     kpCord = (123456789, 123456789)


    if len(good) != 0: #[0] != 123456789 & kpCord[1] != 123456789:
        print(kpCord)
        cv2.circle(gray_frame, kpCord, 20, (255, 0, 255), 5)
        print("hvjcvbngm,nh,cbjk.hgchjkl-hgchjklòhgfhjkò")
        cv2.imshow("img", gray_frame)

    # img3 = cv2.drawMatchesKnn(img1, kp1, gray_frame, kp2, good, None, flags=2)

    # if len(good) > 8:
    #     query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #     train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #     matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 10.0)
    #     matches_mask = mask.ravel().tolist()
    #
    #     h, w = img.shape
    #     pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    #     dst = cv2.perspectiveTransform(pts, matrix)
    #
    #     homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
    #     cv2.imshow("Homography", homography)
    # else:
    #     cv2.imshow("Homography",gray_frame)


    # print(len(good))

    # cv2.imshow("kp1", imgKp1)
    # cv2.imshow("kp2", imgKp2)

    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", gray_frame)
    # cv2.imshow("img3", img3)

    key = cv2.waitKey(1)
    if key == 27:
        break


