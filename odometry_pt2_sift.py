import cv2
import os
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')




# Папка с кадрами
img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"

# Сортировка файлов
images = sorted(os.listdir(img_folder))

# Чтение первого кадра
img_path = os.path.join(img_folder, images[0])
old_frame = cv2.imread(img_path)

# Создание детектора SIFT

max_features = 500 # ограничиваем количество точек которое будет выявлять детектор
sift = cv2.SIFT_create(nfeatures=max_features)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # Метод сопоставления дескрипторов с  cv2.NORM_L2 (евклидово расстояние)
# Параметры камеры
focal_length = 512.0
height, width = old_frame.shape[:2]
principal_point = (width/2, height/2)
K = np.array([[focal_length, 0, principal_point[0]],
              [0, focal_length, principal_point[1]],
              [0, 0, 1]])


trajectory = []

# Начальная позиция камеры
cam_pos = np.zeros((3,1))  # начало координат
cam_scale = 1

for img_name in tqdm(images[1::3]):
    img_path = os.path.join(img_folder, img_name)
    
    frame = cv2.imread(img_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Получаем ключевые точки и дескрипторы
    keypoints_prev, descriptors_prev = sift.detectAndCompute(old_gray, None)
    keypoints_cur, descriptors_cur = sift.detectAndCompute(frame_gray, None)
    
    # При помощи брутфорса находим соответствия между дескрипторами
    matches = bf.match(descriptors_prev, descriptors_cur)
    # Сортируем соответствия по возрастанию расстояния между дескрипторами
    matches = sorted(matches, key=lambda x: x.distance)[:300]

    # Извлекаем координаты ключевых точек, которые соответствуют лучшим матчам
    pts_prev = np.float32([keypoints_prev[m.queryIdx].pt for m in matches])
    pts_curr = np.float32([keypoints_cur[m.trainIdx].pt for m in matches])

    # Находим Essential Matrix с помощью RANSAC для отбрасывания выбросов
    E, mask = cv2.findEssentialMat(
        pts_curr, pts_prev, 
        focal=focal_length, 
        pp=principal_point, 
        method=cv2.RANSAC, 
        prob=0.95, 
        threshold=1.5
    )

    # Берём только inliers из маски RANSAC (корректные соответствия)
    mask = mask.ravel()  # чтобы был одномерный массив 0/1
    pts_prev = pts_prev[mask==1]
    pts_curr = pts_curr[mask==1]

    # Восстанавливаем относительное положение камеры: вращение R и смещение t
    _, R, t, mask_pose = cv2.recoverPose(
        E, pts_curr, pts_prev, 
        focal=focal_length, 
        pp=principal_point
    )
    

    
    # Обновляем глобальную позицию камеры
    cam_pos = cam_pos + cam_scale * t  # cam_scale = масштаб (если известен)
    trajectory.append(cam_pos.flatten())
    old_frame = frame


    
# После обработки всех кадров можно объединить точки
# point_cloud = np.vstack(point_cloud)  # Nx3
trajectory = np.array(trajectory)     # Mx3


# 2D траектория (X-Z)
trajectory_2d = trajectory[:, [0, 2]]

# Визуализация 2D маршрута
plt.figure(figsize=(10,8))
plt.plot(trajectory_2d[:,0], trajectory_2d[:,1], c='red', linewidth=2, label='Маршрут')
plt.scatter(trajectory_2d[:,0], trajectory_2d[:,1], s=1, c='blue')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('2D траектория камеры (монокамера)')
plt.axis('equal')
plt.legend()
plt.show()