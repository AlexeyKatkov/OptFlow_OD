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

def traj_alg(feature_params, lk_params, img_folder):
    # Папка с кадрами


    old_gray = img_folder[0]



    # Параметры камеры
    focal_length = 512.0
    height = 1024
    width = 1024
    principal_point = (width/2, height/2)
    K = np.array([[focal_length, 0, principal_point[0]],
                [0, focal_length, principal_point[1]],
                [0, 0, 1]])

    scale = 0.02

    pos = np.zeros(2, dtype=np.float64)
    trajectory_points = []

    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # shape (N,1,2), dtype=float32

    for frame_gray in tqdm(img_folder[1:]):
        
        

        
        if p0 is None or len(p0) < 50:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)


        # tracking
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if p1 is None:
            # ничего не нашли — переинициализируем и продолжим
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue

        # фильтруем успешные точки
        st = st.reshape(-1)
        good_new = p1[st == 1]
        good_old = p0.reshape(-1, 2)[st == 1]


        good_new = good_new.reshape(-1, 2)
        good_old = good_old.reshape(-1, 2)
        # простой outlier rejection: медианный сдвиг + выбросы по расстоянию
        flow = good_new - good_old   # shape (M, 2)

        if flow.size == 0:  # проверяем, что есть хотя бы одна точка
            # переинициализируем точки и переходим к следующему кадру
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue

        med = np.median(flow, axis=0)
        dx, dy = med[0], med[1]

        # опционально: удалить сильные выбросы, оставить inliers вокруг медианы
        dists = np.linalg.norm(flow - np.array([dx, dy]), axis=1)
        inlier_mask = dists < 5.0   # порог в пикселях — подбирай экспериментально
        if inlier_mask.sum() > 5:
            med_inliers = np.median(flow[inlier_mask], axis=0)
            dx, dy = med_inliers[0], med_inliers[1]

        # накопление положения (2D)

        dy = -dy
        pos += np.array([dx, dy]) * scale   # scale — твой коэффициент (пиксели->метры), см. ниже
        trajectory_points.append(pos.copy())

        # обновляем треки для следующей итерации
        p0 = good_new.reshape(-1, 1, 2).astype(np.float32)
        old_gray = frame_gray.copy()
    
    traj = np.array(trajectory_points)
    gt_time = np.arange(8818)        # индексы GT
    od_time = np.linspace(0, 8817, traj.shape[0])  # индексы твоей траектории

    traj_interp = np.zeros((8818, 2))
    for i in range(2):  # x и y
        f = interp1d(od_time, traj[:, i], kind='linear')
        traj_interp[:, i] = f(gt_time)



    return  traj_interp


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












    

