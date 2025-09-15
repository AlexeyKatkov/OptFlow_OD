import h5py
import numpy as np
import matplotlib.pyplot as plt

#1232321312
#1232321312


# Пути к файлам
file_path = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\sensor_records.hdf5"

# Загружаем данные IMU
with h5py.File(file_path, "r") as f:
    traj_group = f['trajectory_0000']
    imu_group = traj_group['groundtruth']
    
    position_data = np.array(imu_group['position'])  # [x, y, z]

# Разбиваем на X и Y
x = position_data[:,0]
y = position_data[:,1]

# --- Построение траектории XY ---
plt.figure(figsize=(8,8))
plt.plot(x, y, '-o', markersize=2, label="Groundtruth XY")
plt.xlabel("X (метры)")
plt.ylabel("Y (метры)")
plt.title("Траектория дрона (XY) по IMU / groundtruth")
plt.axis('equal')  # Сохраняем пропорции
plt.grid(True)
plt.legend()
plt.show()

# Сохраняем
np.save("C:\\Users\\User\\Desktop\\drones\\opt_dan\\coords\\trajectory.npy", position_data)
print("123")