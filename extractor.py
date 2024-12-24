import cv2
import numpy as np

def orb_detect_and_compute(image, nfeatures=1000):
    """
    ORB로 특징점을 추출하고 디스크립터를 계산합니다.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_descriptors_BF(desc1, desc2):
    """
    ORB 디스크립터를 BFMatcher로 매칭합니다 (Hamming 기준).
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def lucas_kanade_optical_flow(image1, image2, points1):
    """
    Lucas-Kanade 방법으로 Optical Flow를 계산하고,
    추적 성공한 점들만 골라서 반환합니다.
    """
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    points2, status, err = cv2.calcOpticalFlowPyrLK(image1, image2, points1, None, **lk_params)
    
    good_new = points2[status.ravel() == 1]
    good_old = points1[status.ravel() == 1]
    return good_old, good_new

def extract_matched_keypoints(kpts1, kpts2, matches):
    """
    매칭된 keypoints로부터 좌표만 추출하여 반환합니다.
    """
    pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])
    return pts1, pts2