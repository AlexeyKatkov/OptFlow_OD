import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import itertools
img_folder = np.load("C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames_gray.npy")

def AbsoluteTrajectoryError(gt,od):
    return np.sqrt(np.mean( ( (gt - od)**2 ).sum(axis = 1) ) )

def traj_alg(feature_params, lk_params, frames_gray, scale=0.02):
    pos = np.zeros(2, dtype=np.float64)
    trajectory_points = []
    old_gray = frames_gray[0]

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    for frame_gray in tqdm(frames_gray[1:]):
        if p0 is None or len(p0) < 50:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue

        st = st.reshape(-1)
        good_new = p1[st == 1]
        good_old = p0.reshape(-1, 2)[st == 1]
        if good_new.shape[0] == 0:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue

        flow = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)

        if flow.shape[1] != 2:
            flow = flow[:, :2]  # безопасно обрезаем лишнее
        
        if flow.size == 0:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue

        med = np.median(flow, axis=0)
        dx, dy = med[0], med[1]

        dists = np.linalg.norm(flow[:, :2] - med[:2], axis=1)
        inlier_mask = dists < 5.0
        if inlier_mask.sum() > 5:
            med_in = np.median(flow[inlier_mask], axis=0)
            dx, dy = med_in[0], med_in[1]

        dy = -dy
        pos += np.array([dx, dy]) * scale
        trajectory_points.append(pos.copy())
        p0 = good_new.reshape(-1, 1, 2).astype(np.float32)
        old_gray = frame_gray.copy()

    traj = np.array(trajectory_points)
    gt_time = np.arange(8818)
    od_time = np.linspace(0, 8817, traj.shape[0])

    traj_interp = np.zeros((8818, 2))
    for i in range(2):
        f = interp1d(od_time, traj[:, i], kind='linear')
        traj_interp[:, i] = f(gt_time)
    return traj_interp


# ---- Параметры детектора и трекера ----
feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)

lk_params = dict(winSize=(21,21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0, minEigThreshold=1e-4)

# начальные точки для отслеживания



feature_grid = {
    'maxCorners': [200, 500, 1000],
    'qualityLevel': [0.01, 0.03, 0.05],
    'minDistance': [5, 7, 10]
}

lk_grid = {
    'winSize': [(15,15), (21,21), (31,31)],
    'maxLevel': [2, 3, 4],
    'minEigThreshold': [1e-4, 1e-3, 1e-2]
}


gt_traj = np.load("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\trajectory.npy")
gt_traj = gt_traj[:, :2]


best_ate = float('inf')
best_params = None

total_combos = (len(feature_grid['maxCorners']) *
                len(feature_grid['qualityLevel']) *
                len(feature_grid['minDistance']) *
                len(lk_grid['winSize']) *
                len(lk_grid['maxLevel']) *
                len(lk_grid['minEigThreshold']))

progress = tqdm(total=total_combos, desc="Grid Search")

# все комбинации параметров
for fc, ql, md in itertools.product(feature_grid['maxCorners'],
                                   feature_grid['qualityLevel'],
                                   feature_grid['minDistance']):
    for ws, ml, me in itertools.product(lk_grid['winSize'],
                                       lk_grid['maxLevel'],
                                       lk_grid['minEigThreshold']):

        # задаём текущие параметры


        feature_params = dict(maxCorners=fc, qualityLevel=ql, minDistance=md, blockSize=7)
        lk_params = dict(winSize=ws, maxLevel=ml,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                         flags=0, minEigThreshold=me)
        
        # запускаем твой код трекинга на всех кадрах и получаем траекторию
        traj = traj_alg( feature_params, lk_params, img_folder)
        
        # вычисляем метрику
        ate = AbsoluteTrajectoryError(gt_traj, traj)
        
        if ate < best_ate:
            best_ate = ate
            best_params = (feature_params.copy(), lk_params.copy())
            print(f"Новая лучшая ATE={ate:.4f} для {best_params}")
            np.save("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\best_trajectory.npy", traj)

        progress.update(1)

progress.close()












    

