import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Папка с кадрами
img_folder = r"C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"
images = sorted(os.listdir(img_folder))

# Чтение первого кадра
old_frame = cv2.imread(os.path.join(img_folder, images[0]))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Параметры камеры
focal_length = 512.0
height, width = old_frame.shape[:2]
principal_point = (width/2, height/2)
K = np.array([[focal_length, 0, principal_point[0]],
              [0, focal_length, principal_point[1]],
              [0, 0, 1]])

# Инициализация
trajectory = []
cur_pos = np.array([0, 0, 0], dtype=float)
cur_rot = np.eye(3)

orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Настройка графика для динамического обновления
plt.ion()
fig, ax = plt.subplots()
traj_plot, = ax.plot([], [], 'r')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Camera Trajectory')

for img_name in tqdm(images[1::3]):
    frame = cv2.imread(os.path.join(img_folder, img_name))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детекция и дескрипторы
    keypoints_cur = orb.detect(frame_gray, None)
    keypoints_perv = orb.detect(old_gray, None)
    keypoints_cur, descriptors_cur = orb.compute(frame_gray, keypoints_cur)
    keypoints_perv, descriptors_perv = orb.compute(old_gray, keypoints_perv)

    # Сопоставление
    matches = bf.match(descriptors_perv, descriptors_cur)
    matches = sorted(matches, key=lambda x: x.distance)[:200]

    pts_prev = np.float32([keypoints_perv[m.queryIdx].pt for m in matches])
    pts_cur  = np.float32([keypoints_cur[m.trainIdx].pt for m in matches])

    # Essential Matrix и recoverPose
    E, mask = cv2.findEssentialMat(pts_cur, pts_prev, focal=focal_length, pp=principal_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, pts_cur, pts_prev, focal=focal_length, pp=principal_point)

    # Обновление позиции камеры
    cur_pos += cur_rot.dot(t.ravel())
    cur_rot = R.dot(cur_rot)
    trajectory.append(cur_pos.copy())

    # Обновление предыдущего кадра
    old_frame = frame
    old_gray = frame_gray

    # Визуализация текущего кадра
    frame_vis = cv2.drawMatches(old_gray, keypoints_perv, frame_gray, keypoints_cur, matches[:50], None, flags=2)
    cv2.imshow("Frame & Matches", frame_vis)

    # Обновление траектории на графике
    traj_np = np.array(trajectory)
    traj_plot.set_data(traj_np[:,0], traj_np[:,2])
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.001)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
plt.ioff()
plt.show()
