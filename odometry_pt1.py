import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Папка с кадрами
img_folder = r"C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"

# Сортировка файлов
images = sorted(os.listdir(img_folder))

# Чтение первого кадра
img_path = os.path.join(img_folder, images[0])
old_frame = cv2.imread(img_path)

# Параметры для Shi-Tomasi и LK optical flow
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(21,21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Начальная позиция
pos = np.array([0.0, 0.0])
scale = 1 # уменьшенный масштаб для компактной траектории
trajectory_points = [pos.copy()]

for img_name in tqdm(images[1:]):
    img_path = os.path.join(img_folder, img_name)
    frame = cv2.imread(img_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Оптический поток
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None and np.count_nonzero(st) > 0:
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Медианный сдвиг
        flow = good_new - good_old
        dx = float(np.median(flow[:,0]))
        dy = float(np.median(flow[:,1]))

        # Обновляем позицию
        pos += np.array([dx, dy]) * scale
        trajectory_points.append(pos.copy())

        # Переобнаружение точек только если мало фич
        if len(good_new) < 50:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        else:
            p0 = good_new.reshape(-1,1,2).astype(np.float32)
    else:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    old_gray = frame_gray.copy()

# --- Построение графика траектории ---
trajectory_points = np.array(trajectory_points)
plt.figure(figsize=(8,8))
plt.plot(trajectory_points[:,0], trajectory_points[:,1], '-o', markersize=2)
plt.xlabel("X (пиксели)")
plt.ylabel("Y (пиксели)")
plt.title("Визуальная одометрия дрона")
plt.grid(True)
plt.axis('equal')
plt.show()
