import cv2
import os

kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)

if not kamera.isOpened():
    print("Error: Webcam gabisa")
    exit()

save_patch = "home/sorter/dataset/accepted/"
os.makedirs(save_patch, exist_ok=True)
image_count = 0

#tentukan area kotak ROI (x1, y1, x2, y2)
#nanti angka ini diatur dengan disesuikan biji kopi
roi = (200, 100, 440, 380)

print ("Webcam nyala. Tekan 'c' untuk jepret gambar, dan 'q' untuk keluar.")

while True:
    #baca frame dari webcam
    sukses, frame = kamera.read()

    if not sukses:
        print ("Gagal ngambil gambar")
        break

    x1, y1, x2, y2 = roi
    cv2.imshow("Kamera Sorti Kopi - Mode ROI", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        cropped_frame = frame[y1:y2, x1:x2]

        image_path = f"{save_patch}imege_{image_count}.jpg"
        cv2.imwrite(image_path, cropped_frame)
        print(f"Tersimpan: {image_path}")
        image_count += 1

    #jika tombol keluat
    elif key & 0xFF == ord('q'):
        break
    
kamera.release()
cv2.destroyAllWindows()

    