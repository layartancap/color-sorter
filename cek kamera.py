import cv2
import os

kamera = cv2.VideoCapture(0) # Ganti 1 atau 2 kalau pakai webcam USB dan layarnya gelap

# Perbaikan 1: Set Width dan Height yang bener
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not kamera.isOpened():
    print("Error: Webcam gabisa dibuka")
    exit()

# Perbaikan 2: Tambah garis miring (/) di depan 'home'
save_path = "/home/sorter/dataset/accepted/"
os.makedirs(save_path, exist_ok=True)
image_count = 0

# Tentukan area kotak ROI (x1, y1, x2, y2)
# Nanti angka ini diatur dengan disesuaikan ukuran biji kopi di depan kamera
roi = (200, 100, 440, 380)

print("Webcam nyala. Tekan 'c' untuk jepret gambar, dan 'q' untuk keluar.")

while True:
    # Baca frame dari webcam
    sukses, frame = kamera.read()

    if not sukses:
        print("Gagal ngambil gambar")
        break

    x1, y1, x2, y2 = roi
    
    # Perbaikan 3 & 4: Gambar kotak hijau di layar buat panduan naruh kopi
    # Kita pakai .copy() biar kotak hijaunya nggak ikut kesimpen di file foto
    display_frame = frame.copy()
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display_frame, "Taruh Kopi Sini", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Kamera Sortir Kopi - Mode ROI", display_frame)

    key = cv2.waitKey(1)
    
    # Jika tombol 'c' ditekan
    if key & 0xFF == ord('c'):
        # Potong gambar sesuai koordinat kotak ROI
        cropped_frame = frame[y1:y2, x1:x2]

        # Simpan gambar
        image_file = f"{save_path}image_{image_count}.jpg"
        cv2.imwrite(image_file, cropped_frame)
        print(f"Tersimpan: {image_file}")
        image_count += 1

    # Jika tombol 'q' ditekan buat keluar
    elif key & 0xFF == ord('q'):
        break
        
kamera.release()
cv2.destroyAllWindows()
