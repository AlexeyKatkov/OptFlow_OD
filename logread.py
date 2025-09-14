import h5py
import numpy as np

# открыть файл
file_path = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\sensor_records.hdf5"
with h5py.File(file_path, "r") as f:
    # посмотреть корневые группы
    # print(list(f.keys()))  # например ['color', 'imu', 'pose']
    a = np.array(f['trajectory_0000']['camera_data']['color_down'])  # например ['acc', 'gyro', 'timestamps']
print(a)