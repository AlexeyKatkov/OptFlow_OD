import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

img_folder = r"C:\Users\User\Desktop\drones\opt_dan\downloads\my_cutter_res\porabola\res1920x1080"

images = sorted(os.listdir(img_folder))

# Параметры детектора и трекера
feature_params = dict(maxCorners=1500, qualityLevel=0.01, minDistance=10, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=10,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0, minEigThreshold=1e-4)

# Чтение первого кадра
old_frame = cv2.imread(os.path.join(img_folder, images[0]))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

scale = 0.02
alpha = 0.8
pos = np.zeros(2, dtype=np.float64)
prev_pos = np.zeros(2, dtype=np.float64)
trajectory_points = []

for img_name in tqdm(images[1:]):
    frame = cv2.imread(os.path.join(img_folder, img_name))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Переинициализация точек
    if p0 is None or len(p0) < 700:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Оптический поток
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        continue

    st = st.reshape(-1)
    good_new = p1[st == 1].reshape(-1,2)
    good_old = p0.reshape(-1,2)[st == 1].reshape(-1,2)

    if len(good_old) >= 4:  # для гомографии нужно >=4 точек
        H, inliers = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
        if H is not None:
            # извлекаем сдвиг
            dx, dy = H[0,2], H[1,2]

            # грубая оценка угла (берём вращение из верхней 2х2 части матрицы)
            angle = np.arctan2(H[1,0], H[0,0])

            # накопление позиции
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]])
            pos += R @ np.array([dx, dy]) * scale

            # сглаживание
            pos = alpha * pos + (1 - alpha) * prev_pos
            prev_pos = pos
            trajectory_points.append(pos.copy())

    # обновляем треки
    p0 = good_new.reshape(-1,1,2).astype(np.float32)
    old_gray = frame_gray.copy()

# Визуализация
traj = np.array(trajectory_points)
# np.save("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\optflow_trajectory_rot.npy", traj)
# np.save("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\optflow_trajectory_rot1.npy", traj)



plt.figure()
plt.plot(traj[:,0], traj[:,1], '-o', color='blue', markersize=3)
plt.scatter(traj[0,0], traj[0,1], color='green', s=50, label='Start')
plt.scatter(traj[-1,0], traj[-1,1], color='red', s=50, label='End')
plt.title("2D Trajectory with Homography")
plt.legend()
plt.show()
