import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from display import draw_matches, draw_info
from extractor import orb_detect_and_compute, match_descriptors_BF, extract_matched_keypoints
from pointmap import compute_fundamental_matrix, compute_essential_matrix, recover_pose, triangulate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='test_countryroad.mp4',
                        help='Path to video file')
    return parser.parse_args()

def plot_3d_map(points_3d):
    """
    3D 점들을 플롯하는 함수.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, c='blue', label='3D Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Cannot open video:", args.video)
        return

    fx, fy = 718.856, 718.856
    cx, cy = 607.1928, 185.2157
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    ret, frame_old = cap.read()
    if not ret:
        print("Empty video.")
        return

    gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)
    kpts_old, desc_old = orb_detect_and_compute(gray_old, nfeatures=1000)

    # 3D 포인트 저장용 리스트
    all_points_3d = []

    # 첫 카메라 포즈
    pose_old = np.eye(4)

    while True:
        ret, frame_new = cap.read()
        if not ret:
            break
        
        gray_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
        kpts_new, desc_new = orb_detect_and_compute(gray_new, nfeatures=1000)

        if desc_old is not None and desc_new is not None and len(kpts_new) > 0:
            matches = match_descriptors_BF(desc_old, desc_new)
            pts1, pts2 = extract_matched_keypoints(kpts_old, kpts_new, matches)

            if len(pts1) > 8:
                F, inliers = compute_fundamental_matrix(pts1, pts2)
                E = compute_essential_matrix(F, K)

                inlier_pts1 = pts1[inliers.ravel() == 1]
                inlier_pts2 = pts2[inliers.ravel() == 1]
                R, t, mask_pose = recover_pose(E, inlier_pts1, inlier_pts2, K)

                # 현재 프레임의 카메라 포즈 계산
                pose_new = np.eye(4)
                pose_new[:3, :3] = R
                pose_new[:3, 3] = t.ravel()

                # 3D 점 삼각측량
                points_3d_hom = triangulate(pose_old, pose_new, inlier_pts1, inlier_pts2)
                points_3d = points_3d_hom[:, :3] / points_3d_hom[:, 3:]  # Homogeneous -> Euclidean

                # 누적 3D 포인트 추가
                all_points_3d.append(points_3d)

                # ORB 매칭 시각화
                matched_img = draw_matches(frame_old, frame_new, kpts_old, kpts_new, matches)
                cv2.imshow("ORB Matches", matched_img)

                # 현재 포즈를 다음 루프에 전달
                pose_old = pose_new.copy()

        gray_old = gray_new
        kpts_old, desc_old = kpts_new, desc_new
        frame_old = frame_new.copy()

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # 모든 3D 포인트를 하나의 배열로 병합
    all_points_3d = np.vstack(all_points_3d)
    plot_3d_map(all_points_3d)  # 3D 맵 플롯

if __name__ == '__main__':
    main()