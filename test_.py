import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# img_folder = r"C:\Users\User\Desktop\drones\opt_dan\downloads\my_cutter_res\porabola\res1280x720"
img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"

images = sorted(os.listdir(img_folder))

# Параметры трекера
lk_params = dict(winSize=(15,15), maxLevel=10,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0, minEigThreshold=1e-4)

# Параметры визуализации
SIZE = (1024, 1024)
scale = 0.02
theta = 0.0
alpha = 0.8
pos = np.zeros(2, dtype=np.float64)
prev_pos = np.zeros(2, dtype=np.float64)
trajectory_points = []

# Инициализация FAST
fast_detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

# Чтение первого кадра
old_frame = cv2.imread(os.path.join(img_folder, images[0]))
old_frame = cv2.resize(old_frame, SIZE)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Детекция ключевых точек через FAST
keypoints = fast_detector.detect(old_gray, None)
p0 = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else None

for img_name in tqdm(images[1:]):
    frame = cv2.imread(os.path.join(img_folder, img_name))
    frame = cv2.resize(frame, SIZE)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Переинициализация точек, если мало
    if p0 is None or len(p0) < 700:
        keypoints = fast_detector.detect(frame_gray, None)
        p0 = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else None

    # Оптический поток
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None:
        old_gray = frame_gray.copy()
        continue

    st = st.reshape(-1)
    good_new = p1[st==1].reshape(-1,2)
    good_old = p0.reshape(-1,2)[st==1].reshape(-1,2)

    if len(good_old) >= 3:
        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
        if M is not None:
            dx, dy = M[0,2], M[1,2]
            angle = np.arctan2(M[1,0], M[0,0])
            theta += angle
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
            pos += R @ np.array([dx, dy]) * scale
            pos = alpha * pos + (1 - alpha) * prev_pos
            prev_pos = pos
            trajectory_points.append(pos.copy())

    # обновляем треки
    p0 = good_new.reshape(-1,1,2).astype(np.float32)
    old_gray = frame_gray.copy()

# Визуализация траектории
traj = np.array(trajectory_points)
plt.figure()
plt.plot(traj[:,0], traj[:,1], '-o', color='blue', markersize=3)
plt.scatter(traj[0,0], traj[0,1], color='green', s=50, label='Start')
plt.scatter(traj[-1,0], traj[-1,1], color='red', s=50, label='End')
plt.title("2D Trajectory with Rotation Compensation (FAST)")
plt.legend()
plt.show()
