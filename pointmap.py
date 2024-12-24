import numpy as np
import cv2

def add_ones(pts):
    """
    2D 배열의 점에 1을 추가하여 (x, y, 1) 형태의
    homogeneous 좌표로 확장합니다.
    """
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def compute_fundamental_matrix(pts1, pts2):
    """
    RANSAC으로 Fundamental Matrix를 계산합니다.
    """
    F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, inliers

def compute_essential_matrix(F, K):
    """
    카메라 내부 파라미터(K)로부터 Essential Matrix를 계산합니다.
    E = K^T * F * K
    """
    return K.T @ F @ K

def recover_pose(E, pts1, pts2, K):
    """
    Essential Matrix를 통해 R, t를 복원합니다.
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask

def triangulate(pose1, pose2, pts1, pts2):
    """
    두 카메라 Pose와 매칭된 점들로부터 3D 위치를
    삼각측량합니다.
    pose1, pose2는 각각 4x4 형태의 RT 행렬(또는 그 역행렬)이라 가정.
    """
    ret = np.zeros((pts1.shape[0], 4))

    # pose를 invert해서 Projection Matrix 형태로 사용 (간단히 예시)
    pose1_inv = np.linalg.inv(pose1)
    pose2_inv = np.linalg.inv(pose2)

    for i, (p1, p2) in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p1[0] * pose1_inv[2] - pose1_inv[0]
        A[1] = p1[1] * pose1_inv[2] - pose1_inv[1]
        A[2] = p2[0] * pose2_inv[2] - pose2_inv[0]
        A[3] = p2[1] * pose2_inv[2] - pose2_inv[1]

        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret  # 3D 포인트(동차 좌표)

def vertex_to_points(optimizer, first_point_id, last_point_id):
    """
    g2o 등으로 최적화된 3D 포인트 vertices를 (x,y,z)로 변환.
    여기는 예시 코드이므로 실제 optimizer 세팅은 별도 구성 필요.
    """
    vertices_dict = optimizer.vertices()
    estimated_points = []
    for idx in range(first_point_id, last_point_id):
        estimated_points.append(vertices_dict[idx].estimate())
    estimated_points = np.array(estimated_points)
    xs, ys, zs = estimated_points.T
    return xs, ys, zs

def compute_reprojection_error(intrinsics, extrinsics, points_3d, observations, project_func):
    """
    특정 카메라 파라미터/포즈/3D 점들이 주어졌을 때,
    reprojection error를 계산하는 예시 함수입니다.
    project_func: 3D점을 2D로 사영하는 함수 (별도 정의)
    """
    total_error = 0
    num_points = 0
    
    for (rotation, translation), obs_2d in zip(extrinsics, observations):
        # 3D -> 2D
        proj_points = project_func(points_3d, intrinsics, rotation, translation)
        error = np.linalg.norm(proj_points - obs_2d, axis=1)
        total_error += np.sum(error)
        num_points += len(points_3d)
    
    mean_error = total_error / num_points
    return mean_error