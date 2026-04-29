"""
Real-time machine learning sorter for Raspberry Pi 4 with solenoid output.

Run this after preparing data with preprocess.py and training with train.py.
The solenoid relay follows the wiring in solenid.py:
- GPIO 18 using BCM numbering
- Active LOW relay: GPIO LOW turns solenoid ON, GPIO HIGH turns it OFF
"""

from time import monotonic, sleep
import pickle

import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO


MODEL_PATH = "defect_model.pkl"
SOLENOID_PIN = 18
SOLENOID_ACTIVE_LOW = True
SOLENOID_PULSE_SECONDS = 0.25
SOLENOID_COOLDOWN_SECONDS = 0.8

CAMERA_SIZE = (640, 480)
ROI = (200, 100, 440, 380)

# HSV range for detecting the object before classification.
# Adjust these values to match the object color and lighting.
LOWER_ORANGE = np.array([10, 100, 100])
UPPER_ORANGE = np.array([25, 255, 255])

LABELS = {
    0: "Defect-Free",
    1: "Defective",
    2: "Neutral",
}

LABEL_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 255, 0),
}


class SolenoidController:
    def __init__(self, pin, active_low=True):
        self.pin = pin
        self.active_low = active_low
        self.off_level = GPIO.HIGH if active_low else GPIO.LOW
        self.on_level = GPIO.LOW if active_low else GPIO.HIGH
        self.off_at = 0.0
        self.last_triggered_at = 0.0

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=self.off_level)

    def trigger(self, duration=SOLENOID_PULSE_SECONDS, cooldown=SOLENOID_COOLDOWN_SECONDS):
        now = monotonic()
        if now - self.last_triggered_at < cooldown:
            return False

        GPIO.output(self.pin, self.on_level)
        self.off_at = now + duration
        self.last_triggered_at = now
        return True

    def update(self):
        if self.off_at and monotonic() >= self.off_at:
            self.off()

    def off(self):
        GPIO.output(self.pin, self.off_level)
        self.off_at = 0.0

    def cleanup(self):
        self.off()
        GPIO.cleanup(self.pin)


def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Error: {path} tidak ditemukan. Jalankan train.py terlebih dahulu."
        ) from exc


def preprocess_image(image):
    resized = cv2.resize(image, (64, 64))
    return resized.flatten()


def is_object_present_and_centered(roi_frame, lower_color, upper_color):
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    detected_pixels = cv2.countNonZero(mask)

    if detected_pixels <= 500:
        return False

    moments = cv2.moments(mask)
    if moments["m00"] <= 0:
        return False

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    height, width = mask.shape

    return abs(cx - width // 2) < 30 and abs(cy - height // 2) < 30


def draw_status(frame, counts, solenoid_active):
    cv2.putText(
        frame,
        f"Good Quality: {counts[0]}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        LABEL_COLORS[0],
        2,
    )
    cv2.putText(
        frame,
        f"Defective: {counts[1]}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        LABEL_COLORS[1],
        2,
    )
    cv2.putText(
        frame,
        f"Neutral: {counts[2]}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        LABEL_COLORS[2],
        2,
    )
    cv2.putText(
        frame,
        f"Solenoid: {'ON' if solenoid_active else 'OFF'}",
        (10, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def main():
    model = load_model(MODEL_PATH)
    solenoid = SolenoidController(SOLENOID_PIN, SOLENOID_ACTIVE_LOW)
    picam2 = Picamera2()
    counts = {0: 0, 1: 0, 2: 0}
    object_in_roi = False

    try:
        picam2.configure(
            picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": CAMERA_SIZE}
            )
        )
        picam2.start()
        sleep(1)

        print("Model loaded. Tekan 'q' untuk keluar.")

        while True:
            solenoid.update()

            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            x1, y1, x2, y2 = ROI
            color = (255, 255, 0)
            label_text = "Waiting for object"
            roi_frame = frame[y1:y2, x1:x2]

            object_ready = is_object_present_and_centered(
                roi_frame,
                LOWER_ORANGE,
                UPPER_ORANGE,
            )

            if object_ready:
                processed = preprocess_image(roi_frame)
                prediction = int(model.predict([processed])[0])
                label_text = LABELS.get(prediction, f"Unknown {prediction}")
                color = LABEL_COLORS.get(prediction, (255, 255, 255))

                if not object_in_roi:
                    counts[prediction] = counts.get(prediction, 0) + 1
                    object_in_roi = True

                    if prediction == 1:
                        solenoid.trigger()
                        print("Defective detected: solenoid triggered.")
                    else:
                        print(f"{label_text} detected: solenoid not triggered.")
            else:
                object_in_roi = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Prediction: {label_text}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            draw_status(frame, counts, solenoid.off_at > 0)
            cv2.imshow("ML Solenoid Sorter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        solenoid.cleanup()


if __name__ == "__main__":
    main()
