import cv2
import numpy as np

in_path = r'D:\Yun\sea_trash\train_data - 複製\images\20180924_花蓮溪口南岸_壽豐-林東良攝-IMG_7728.jpg'
out_path = r'C:\Users\Chen_ZY\yolov7_sea\runs\20180924_花蓮溪口南岸_壽豐-林東良攝-IMG_7728.jpg'


## Chinese cv2.imread()

# img = cv2.imread(out_path)

data_path = open(in_path,"rb")
bytes = bytearray(data_path.read())
numpyarray = np.asarray(bytes, dtype = np.uint8)
img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)


## Chinese cv2.imwrite()

# cv2.imwrite(out_path, img)

cv2.imencode('.jpg', img)[1].tofile(out_path)
