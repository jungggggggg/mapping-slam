import cv2

def draw_keypoints(frame, keypoints):
    """
    ORB, etc.로 추출된 keypoint들을 프레임 위에 시각화합니다.
    """
    return cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=0)

def draw_matches(frame1, frame2, kpts1, kpts2, matches):
    """
    두 프레임 간 매칭 결과를 시각화합니다.
    """
    matched_image = cv2.drawMatches(frame1, kpts1, frame2, kpts2, matches[:50], None, flags=2)
    return matched_image

def draw_info(frame, text, pos=(10,30)):
    """
    프레임에 임의의 텍스트 정보를 표시합니다.
    """
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)