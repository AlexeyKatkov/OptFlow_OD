# import cv2
# import os
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt






# # Папка с кадрами
# img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"

# # Сортировка файлов
# images = sorted(os.listdir(img_folder))

# # Чтение первого кадра
# img_path = os.path.join(img_folder, images[0])
# old_frame = cv2.imread(img_path)
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)



# # Параметры камеры
# focal_length = 512.0
# height, width = old_frame.shape[:2]
# principal_point = (width/2, height/2)
# K = np.array([[focal_length, 0, principal_point[0]],
#               [0, focal_length, principal_point[1]],
#               [0, 0, 1]])



# # ---- Параметры детектора и трекера ---- После гридсерча
# feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=5, blockSize=7)
# lk_params = dict(winSize=(15,15), maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
#                  flags=0, minEigThreshold=1e-4)

# # feature_params = dict(maxCorners=500, qualityLevel=0.5, minDistance=7, blockSize=7)
# # lk_params = dict(winSize=(21,21), maxLevel=3,
# #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
# #                  flags=0, minEigThreshold=1e-4)

# # начальные точки для отслеживания

# scale = 0.02

# pos = np.zeros(2, dtype=np.float64)
# trajectory_points = []

# p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # shape (N,1,2), dtype=float32

# for img_name in tqdm(images[1:]):
#     img_path = os.path.join(img_folder, img_name)
    
#     frame = cv2.imread(img_path)
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
#     if p0 is None or len(p0) < 50:
#         p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)


#     # tracking
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     if p1 is None:
#         # ничего не нашли — переинициализируем и продолжим
#         p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
#         old_gray = frame_gray.copy()
#         continue

#     # фильтруем успешные точки
#     st = st.reshape(-1)
#     good_new = p1[st == 1]
#     good_old = p0.reshape(-1, 2)[st == 1]


#     good_new = good_new.reshape(-1, 2)
#     good_old = good_old.reshape(-1, 2)
#     # простой outlier rejection: медианный сдвиг + выбросы по расстоянию
#     flow = good_new - good_old   # shape (M, 2)

#     if flow.size == 0:  # проверяем, что есть хотя бы одна точка
#         # переинициализируем точки и переходим к следующему кадру
#         p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
#         old_gray = frame_gray.copy()
#         continue

#     med = np.median(flow, axis=0)
#     dx, dy = med[0], med[1]

#     # опционально: удалить сильные выбросы, оставить inliers вокруг медианы
#     dists = np.linalg.norm(flow - np.array([dx, dy]), axis=1)
#     inlier_mask = dists < 5.0   # порог в пикселях — подбирай экспериментально
#     if inlier_mask.sum() > 5:
#         med_inliers = np.median(flow[inlier_mask], axis=0)
#         dx, dy = med_inliers[0], med_inliers[1]

#     # накопление положения (2D)

#     dx = -dx
#     pos += np.array([dx, dy]) * scale   # scale — твой коэффициент (пиксели->метры), см. ниже
#     trajectory_points.append(pos.copy())

#     # обновляем треки для следующей итерации
#     p0 = good_new.reshape(-1, 1, 2).astype(np.float32)
#     old_gray = frame_gray.copy()




# traj = np.array(trajectory_points)

# # Сохраняем
# # np.save("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\optflow_trajectory.npy", trajectory_points)
# plt.figure()

# # Основная траектория
# plt.plot(traj[:,0], traj[:,1], '-o', color='blue', markersize=3)

# # Начальная точка — зелёная
# plt.scatter(traj[0,0], traj[0,1], color='green', s=50, label='Start')

# # Конечная точка — красная
# plt.scatter(traj[-1,0], traj[-1,1], color='red', s=50, label='End')

# # plt.gca().invert_yaxis()  # чтобы направление Y было привычное
# plt.title("2D Trajectory")
# plt.legend()
# plt.show()

    




































import cv2
import os
import numpy as np

# Папка с кадрами
img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"
images = sorted(os.listdir(img_folder))

# Параметры детектора и трекера
feature_params = dict(maxCorners=2000, qualityLevel=0.1, minDistance=2, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0, minEigThreshold=1e-4)

# Чтение первого кадра
frame_idx = 0
old_frame = cv2.imread(os.path.join(img_folder, images[frame_idx]))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Изначально точки через Shi-Tomasi
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cv2.namedWindow("Optical Flow Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Optical Flow Viewer", 800, 600)

while True:
    # Чтение следующего кадра
    frame_idx = min(frame_idx, len(images)-1)
    frame_path = os.path.join(img_folder, images[frame_idx])
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Если точек мало или нет → сначала Shi-Tomasi, потом FAST
    if p0 is None or len(p0) < 700:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        # if p0 is None or len(p0) <250:
        #     fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        #     keypoints = fast.detect(frame_gray, None)

        #     max_points = 1000
        #     keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)  # сортируем по "силе"
        #     keypoints = keypoints[:max_points]
        #     if keypoints:
        #         p0 = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1,1,2)
        #     else:
        #         p0 = p0 if p0 is not None else None

    # Вычисляем оптический поток
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    else:
        p1 = None

    mask = np.zeros_like(frame)
    num_points = 0  # для отображения

    if p1 is not None:
        st = st.reshape(-1)
        good_new = p1[st==1].reshape(-1,2)
        good_old = p0.reshape(-1,2)[st==1].reshape(-1,2)
        num_points = len(good_new)

        # Рисуем стрелки
        for (new, old_pt) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old_pt.ravel()
            mask = cv2.arrowedLine(mask, (int(c), int(d)), (int(a), int(b)),
                                   color=(255,0,255), thickness=2, tipLength=0.3)
        img = cv2.add(frame, mask)
    else:
        img = frame

    # Выводим количество точек в правом верхнем углу
    cv2.putText(img, f'Points: {num_points}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Optical Flow Viewer", img)

    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == ord('a'):  # влево
        frame_idx = max(0, frame_idx - 1)
    elif key == ord('d'):  # вправо
        frame_idx = min(len(images)-1, frame_idx + 1)

    # Обновляем предыдущий кадр и точки
    old_gray = frame_gray.copy()
    if p1 is not None and len(good_new) > 0:
        p0 = good_new.reshape(-1,1,2).astype(np.float32)

cv2.destroyAllWindows()
