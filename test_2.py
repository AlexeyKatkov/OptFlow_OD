import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# img_folder = "/media/nil-risu/Files/gridsearch/0000/frames"
img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"
# img_folder = r"C:\Users\User\Desktop\drones\opt_dan\downloads\my_cutter_res\diagonal_target_frames_step25_size_1024"
# img_folder = r"C:\Users\User\Desktop\drones\opt_dan\downloads\my_cutter_res\porabola\res1280x720"
# img_folder = r"C:\Users\User\Desktop\drones\opt_dan\downloads\my_cutter_res\goldencut\res1280x720"
images = sorted(os.listdir(img_folder))

# Параметры детектора и трекера
feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=5, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=10,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0, minEigThreshold=1e-4)

# Чтение первого кадра
frame_idx = 0
SIZE = (1024, 1024)
old_frame = cv2.imread(os.path.join(img_folder, images[0]))
old_frame = cv2.resize(old_frame, SIZE)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

scale = 0.02
theta = 0.0
alpha = 0.8
pos = np.zeros(2, dtype=np.float64)
prev_pos = np.zeros(2, dtype=np.float64)
trajectory_points = []

for img_name in tqdm(images[1:]):
    frame = cv2.imread(os.path.join(img_folder, img_name))
    frame = cv2.resize(frame, SIZE)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Переинициализация точек, если мало
    if p0 is None or len(p0) < 700:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Оптический поток
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        continue

    st = st.reshape(-1)
    good_new = p1[st==1].reshape(-1,2)
    good_old = p0.reshape(-1,2)[st==1].reshape(-1,2)

    if len(good_old) >= 3:
        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
        if M is not None:
            # translation
            dx, dy = M[0,2], M[1,2]
            # rotation
            angle = np.arctan2(M[1,0], M[0,0])
            theta += angle
            # обновляем позицию с учётом поворота
            R = np.array( [[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]] )
            

            pos += R @ np.array([dx, dy]) * scale
           
            pos = alpha * pos + (1 - alpha) * prev_pos
            prev_pos = pos
            trajectory_points.append(pos.copy())

    # обновляем треки
    p0 = good_new.reshape(-1,1,2).astype(np.float32)
    old_gray = frame_gray.copy()

# Визуализация траектории
traj = np.array(trajectory_points)
# np.save("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\optflow_trajectory_rot.npy", traj)
# np.save("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\optflow_trajectory_rot1.npy", traj)

plt.figure()
plt.plot(traj[:,0], traj[:,1], '-o', color='blue', markersize=3)
plt.scatter(traj[0,0], traj[0,1], color='green', s=50, label='Start')
plt.scatter(traj[-1,0], traj[-1,1], color='red', s=50, label='End')
plt.title("2D Trajectory with Rotation Compensation")
plt.legend()
plt.show()
