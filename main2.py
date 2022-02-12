import cv2
import numpy as np

img = cv2.imread('copertina.jpg', cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture(0)

# Features(sift algorithm)

# sift = cv2.SIFT_create()
# kp_image, desc_image = sift.detectAndCompute(img,
#                                              None)  # sift needs key points and descriptors to work for the image and the frame
# img = cv2.drawKeypoints(img, kp_image, img)                             #query image        #finds keypoints on the img

orb = cv2.ORB_create()
kp_img_orb, desc_img_orb = orb.detectAndCompute(img, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray scale

    # kp_grayframe, desc_grayframe = sift.detectAndCompute(gray_frame, None)

    kp_frame_orb, desc_frame_orb = orb.detectAndCompute(gray_frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_img_orb, desc_frame_orb)
    matches = sorted(matches, key=lambda x: x.distance)

    matching_result = cv2.drawMatches(img, kp_img_orb, gray_frame, kp_frame_orb, matches[:50], None, flags=2)

    matches = flann.knnMatch(desc_img_orb, desc_frame_orb, k=2)

    good_points = []
    for m in matches:
       if m.distance < 0.6:                 ###*****RIPRENDI DA QUI**********
           good_points.append(m)

    #img3 = cv2.drawMatches(img, kp_img_orb, gray_frame, kp_frame_orb, good_points, gray_frame)

    # Homography
    if len(good_points) > 10:                   #at least 10 matches to draw homography
        query_pts = np.float32([kp_img_orb[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_frame_orb[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        # Perspective transform
        h, w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", gray_frame)


    cv2.imshow("Matching result", matching_result)
    #cv2.imshow('Img3', img3)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()

