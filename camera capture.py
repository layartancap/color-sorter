"""
DESCRIPTION:
this code to capture images from the Raspberry Pi Camera

AUTHOR   : Solehin Rizal
WEBSITE  : www.cytron.io
EMAIL    : solehin@cytron.io
"""

import cv2
import os
from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

save_path = "/home/sorter/dataset/"
# Bikin foldernya otomatis kalau belum ada biar nggak error
os.makedirs(save_path, exist_ok=True) 

# Logika pinter buat nerusin nomor urut gambar
image_count = 0
while os.path.exists(f"{save_path}image_{image_count}.jpg"):
    image_count += 1

roi = (200, 100, 440, 380)

print("Tekan 'c' untuk jepret, dan 'q' untuk keluar.")
print(f"Sistem siap. Gambar berikutnya akan disimpan sebagai: image_{image_count}.jpg")

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    x1, y1, x2, y2 = roi
    
    # Pisahkan gambar layar dengan gambar asli agar kotak biru tidak ikut terfoto
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Capturing Images", display_frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        cropped_frame = frame[y1:y2, x1:x2]
        image_path = f"{save_path}image_{image_count}.jpg"
        cv2.imwrite(image_path, cropped_frame)
        print(f"Tersimpan: {image_path}")
        image_count += 1
    elif key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
