import h5py
import numpy as np
import cv2
import os

# Пути к файлам
file_path = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\sensor_records.hdf5"
img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"

# Загружаем данные IMU
with h5py.File(file_path, "r") as f:
    traj_group = f['trajectory_0000']
    imu_group = traj_group['groundtruth']
    
    gyro_data = np.array(imu_group['gyroscope'])  # Гироскоп [x, y, z]
    accel_data = np.array(imu_group['accelerometer'])  # Акселерометр [x, y, z]
    
    timestamps = np.arange(len(gyro_data)) / 100.0  # t = k/100

# Получаем и сортируем кадры
images = os.listdir(img_folder)
total_frames = len(images)

print(f"Всего кадров: {total_frames}")
print(f"Всего замеров IMU: {len(gyro_data)}")
print(f"Длительность записи: {timestamps[-1]:.2f} секунд")

# Определяем частоту кадров видео
if total_frames > 1 and timestamps[-1] > 0:
    video_fps = total_frames / timestamps[-1]
    print(f"Примерная частота видео: {video_fps:.2f} FPS")
else:
    video_fps = 25.0  # Значение по умолчанию

for frame_idx, img_name in enumerate(images):
    img_path = os.path.join(img_folder, img_name)
    frame = cv2.imread(img_path)
    
    if frame is None:
        continue
    
    # Вычисляем время для текущего кадра
    current_time = frame_idx / video_fps
    
    # Находим ближайший замер IMU
    imu_idx = int(current_time * 100)  # т.к. IMU на 100 Гц
    imu_idx = min(imu_idx, len(gyro_data) - 1)  # Защита от выхода за границы
    
    # Получаем текущие данные IMU
    current_gyro = gyro_data[imu_idx]
    current_accel = accel_data[imu_idx]
    
    # Создаем копию кадра для наложения текста
    display_frame = frame.copy()
    
    # Добавляем информацию на кадр
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)  # Зеленый
    line_type = 2
    
    y_offset = 30
    line_height = 30
    
    # Время и номер кадра
    cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames}", 
                (10, y_offset), font, font_scale, font_color, line_type)
    cv2.putText(display_frame, f"Time: {current_time:.2f}s", 
                (10, y_offset + line_height), font, font_scale, font_color, line_type)
    
    # Данные гироскопа
    cv2.putText(display_frame, "Gyroscope (rad/s):", 
                (10, y_offset + line_height * 2), font, font_scale, font_color, line_type)
    cv2.putText(display_frame, f"  X: {current_gyro[0]:.4f}", 
                (10, y_offset + line_height * 3), font, font_scale, font_color, line_type)
    cv2.putText(display_frame, f"  Y: {current_gyro[1]:.4f}", 
                (10, y_offset + line_height * 4), font, font_scale, font_color, line_type)
    cv2.putText(display_frame, f"  Z: {current_gyro[2]:.4f}", 
                (10, y_offset + line_height * 5), font, font_scale, font_color, line_type)
    
    # Данные акселерометра
    cv2.putText(display_frame, "Accelerometer (m/s²):", 
                (10, y_offset + line_height * 6), font, font_scale, font_color, line_type)
    cv2.putText(display_frame, f"  X: {current_accel[0]:.4f}", 
                (10, y_offset + line_height * 7), font, font_scale, font_color, line_type)
    cv2.putText(display_frame, f"  Y: {current_accel[1]:.4f}", 
                (10, y_offset + line_height * 8), font, font_scale, font_color, line_type)
    cv2.putText(display_frame, f"  Z: {current_accel[2]:.4f}", 
                (10, y_offset + line_height * 9), font, font_scale, font_color, line_type)
    
    # Показываем кадр с данными
    cv2.imshow("Video with IMU Data", display_frame)
    
    # Управление скоростью воспроизведения
    delay = int(1000 / video_fps)  # Задержка для правильной скорости
    key = cv2.waitKey(delay) & 0xFF
    
    if key == ord('q'):  # Выход по 'q'
        break
    elif key == ord('p'):  # Пауза по 'p'
        while True:
            key2 = cv2.waitKey(1)
            if key2 == ord('p') or key2 == ord('q'):
                break
        if key2 == ord('q'):
            break

cv2.destroyAllWindows()