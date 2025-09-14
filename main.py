import cv2
import os

img_folder = "C:\\Users\\User\\Desktop\\drones\\opt_dan\\downloads\\0000\\frames"

# Получаем список файлов и сортируем по имени
images = os.listdir(img_folder)
# print(images)
for img_name in images:
    img_path = os.path.join(img_folder, img_name)
    frame = cv2.imread(img_path)
    
    cv2.imshow("Video Preview", frame)
    
    # Ждём 100 мс между кадрами (10 fps)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # можно нажать q чтобы выйти
        break

cv2.destroyAllWindows()
print("hhh")
