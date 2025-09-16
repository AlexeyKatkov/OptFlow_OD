import cv2
import os
import numpy as np

# Папка с кадрами
img_folder = r"/media/nil-risu/Files/gridsearch/0000/frames"
images = sorted(os.listdir(img_folder))

# Параметры детектора и трекера
feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=5, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0, minEigThreshold=1e-4)

scale = 0.02
pos = np.zeros(2, dtype=np.float64)
trajectory_points = []

# Начальные кадры
idx = 0
old_frame = cv2.imread(os.path.join(img_folder, images[idx]))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    frame = old_frame.copy()

    # Рисуем траекторию
    for point in trajectory_points:
        cv2.circle(frame, tuple(point.astype(int)), 2, (255,0,0), -1)

    cv2.imshow("Optical Flow", frame)
    key = cv2.waitKey(0)  # ждём нажатия клавиши

    if key == 27:  # ESC — выход
        break
    elif key == 81:  # ←
        idx = max(0, idx - 1)
    elif key == 83:  # →
        idx = min(len(images)-1, idx + 1)
    else:
        continue  # игнорируем другие клавиши

    # Загружаем следующий кадр
    new_frame = cv2.imread(os.path.join(img_folder, images[idx]))
    frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    if p0 is None or len(p0) < 50:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Вычисляем оптический поток
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0.reshape(-1,2)[st==1]
        flow = good_new - good_old
        if flow.size > 0:
            med = np.median(flow, axis=0)
            dx, dy = -med[0], med[1]
            pos += np.array([dx, dy]) * scale
            trajectory_points.append(pos.copy())

        # обновляем треки
        p0 = good_new.reshape(-1,1,2).astype(np.float32)

    old_gray = frame_gray.copy()
    old_frame = new_frame.copy()

cv2.destroyAllWindows()
