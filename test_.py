import cv2
import os
import numpy as np
import time

# Папка с кадрами
# img_folder = "/media/nil-risu/Files/gridsearch/0000/frames"
img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"
images = sorted(os.listdir(img_folder))

# Параметры детектора и трекера
feature_params = dict(maxCorners=2000, qualityLevel=0.1, minDistance=2, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0, minEigThreshold=1e-4)

SIZE = (1024,1024)
frame_idx = 0
old_frame = cv2.imread(os.path.join(img_folder, images[frame_idx]))
old_frame = cv2.resize(old_frame, SIZE)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Изначальные точки через Shi-Tomasi
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

cv2.namedWindow("Optical Flow Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Optical Flow Viewer", 800, 600)

# Изначальная линия (направление камеры)
origin = np.array([320, 240], dtype=np.float32)
line_length = 50
initial_line = np.array([origin, origin + np.array([line_length, 0])], dtype=np.float32)  # горизонтально вправо
line_global = initial_line.copy()  # будет хранить поворот относительно старта

theta_total = 0.0  # накопленный угол

while True:
    frame_idx = min(frame_idx, len(images)-1)
    frame_path = os.path.join(img_folder, images[frame_idx])
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, SIZE)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is None or len(p0) < 700:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    else:
        p1 = None

    mask = np.zeros_like(frame)
    num_points = 0

    if p1 is not None:
        st = st.reshape(-1)
        good_new = p1[st==1].reshape(-1,2)
        good_old = p0.reshape(-1,2)[st==1].reshape(-1,2)
        num_points = len(good_new)

        if len(good_old) >= 3:
            M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
            if M is not None:
                # только вращение и масштаб (translation игнорируем для линии)
                angle = np.arctan2(M[1,0], M[0,0])
                theta_total += angle
                R = np.array([[np.cos(theta_total), -np.sin(theta_total)],
                              [np.sin(theta_total),  np.cos(theta_total)]], dtype=np.float32)

                # применяем вращение к изначальной линии
                line_global = (R @ (initial_line - origin).T).T + origin

        # рисуем линии опт. потока
        for (new, old_pt) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old_pt.ravel()
            mask = cv2.arrowedLine(mask, (int(c), int(d)), (int(a), int(b)),
                                   color=(255,0,255), thickness=2, tipLength=0.3)

        # рисуем “компас” камеры
        cv2.line(mask, tuple(line_global[0].astype(int)), tuple(line_global[1].astype(int)), (0,255,0), 3)

        img = cv2.add(frame, mask)
    else:
        img = frame

    cv2.putText(img, f'Points: {num_points}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Optical Flow Viewer", img)

    key = cv2.waitKey(1)
    if key == 27:
        break
    # elif key == ord('a'):
    #     frame_idx = max(0, frame_idx - 1)
    # elif key == ord('d'):
    time.sleep(0.01)
    frame_idx = min(len(images)-1, frame_idx + 1)

    old_gray = frame_gray.copy()
    if p1 is not None and len(good_new) > 0:
        p0 = good_new.reshape(-1,1,2).astype(np.float32)

cv2.destroyAllWindows()
