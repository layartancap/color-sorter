"""
DESCRIPTION:
this code to test the trained model in real-time

AUTHOR   : Solehin Rizal
WEBSITE  : www.cytron.io
EMAIL    : solehin@cytron.io
"""


import cv2
import numpy as np
import pickle
from picamera2 import Picamera2

# Load the trained model
try:
    with open("defect_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: defect_model.pkl not found. Train the model first.")
    exit()

def preprocess_image(image):
    """
    Resize the image to 64x64 and flatten it for model prediction.
    """
    resized = cv2.resize(image, (64, 64))
    flattened = resized.flatten()
    return flattened

def is_object_present_and_centered(roi_frame, lower_color, upper_color):
    """
    Check if an object is present in the ROI and centered.
    Uses color masking to detect the object and ensures it is near the center.
    """
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    detected_pixels = cv2.countNonZero(mask)

    # Check if enough pixels are detected (object is present)
    if detected_pixels > 500:  # Adjust threshold as needed
        # Calculate the moments of the mask to find the center
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])  # X center of the object
            cy = int(moments["m01"] / moments["m00"])  # Y center of the object
            height, width = mask.shape

            # Check if the object's center is close to the ROI's center
            if abs(cx - width // 2) < 30 and abs(cy - height // 2) < 30:  # 30-pixel tolerance
                return True
    return False

# Initialize the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Define ROI (Region of Interest)
roi = (200, 100, 440, 380)  # Adjust as needed

# Define color range for object detection (e.g., orange ping pong balls)
lower_orange = np.array([10, 100, 100])  # Adjust HSV values as needed
upper_orange = np.array([25, 255, 255])  # Adjust HSV values as needed

# Initialize counters for each category
count_defect_free = 0
count_defective = 0
count_neutral = 0

# Flag to ensure an object is counted only once
object_in_roi = False

print("Press 'q' to quit.")

while True:
    # Capture a frame
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Draw ROI rectangle
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow box for ROI

    # Crop the ROI for processing
    roi_frame = frame[y1:y2, x1:x2]

    # Check if an object is present and centered in the ROI
    if is_object_present_and_centered(roi_frame, lower_orange, upper_orange):
        # Preprocess the ROI frame for classification
        processed = preprocess_image(roi_frame)

        # Perform the prediction
        prediction = model.predict([processed])[0]
        confidence = model.predict_proba([processed])[0]  # Get confidence scores

        # Map prediction to labels
        labels_map = {0: "Defect-Free", 1: "Defective", 2: "Neutral"}
        label = labels_map[prediction]

        # Assign colors based on class
        colors_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 255, 0)}  # Green, Red, Yellow
        color = colors_map[prediction]

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Prediction: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Count objects passing through the ROI
        if not object_in_roi:
            if prediction == 0:  # Defect-Free
                count_defect_free += 1
            elif prediction == 1:  # Defective
                count_defective += 1
            elif prediction == 2:  # Neutral
                count_neutral += 1
            object_in_roi = True  # Mark object as counted
    else:
        # Reset flag if no object detected or not centered
        object_in_roi = False

    # Display counters on the frame
    cv2.putText(frame, f"Good Quality: {count_defect_free}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Defective: {count_defective}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the live feed with prediction and counters
    cv2.imshow("Defect Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
