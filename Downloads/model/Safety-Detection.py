from ultralytics import YOLO
import cv2
import cvzone
import math
import winsound
from time import time

def detect_with_camera():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Load YOLO model
    model = YOLO(r"the model full train\weights\best.pt")

    # Class names
    classNames = ['Boot', 'Face Protector', 'Gloves', 'Helmet', 'Normal-Glasses', 'Safety-Glasses', 'Vest']

    # Required safety equipment
    REQUIRED_SAFETY_ITEMS = ['Helmet', 'Vest', 'Safety-Glasses', 'Gloves' , 'Boot']  # Modify as needed
    
    # Buzzer control variables
    last_beep_time = 0
    beep_interval = 0.1  # seconds between beeps
    beep_duration = 500  # milliseconds
    beep_frequency = 800  # Hz

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        results = model(img, stream=True)
        
        detected_items = set()
        missing_items = set(REQUIRED_SAFETY_ITEMS)
        safety_alert = False
        non_safety_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.3:  # Only consider confident detections
                    detected_items.add(currentClass)
                    
                    # Update missing items
                    if currentClass in missing_items:
                        missing_items.remove(currentClass)
                    
                    # Check for non-safety items
                    if currentClass == 'Normal-Glasses':
                        non_safety_detected = True
                    
                    # Set box color
                    if currentClass in REQUIRED_SAFETY_ITEMS:
                        myColor = (0, 255, 0)  # Green for proper safety equipment
                    elif currentClass == 'Normal-Glasses':
                        myColor = (255, 0, 0)  # Red for non-safety items
                    else:
                        myColor = (255, 165, 0)  # Orange for other detected items

                    # Draw bounding box and label
                    cvzone.putTextRect(img, f'{currentClass} {conf}',
                                     (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                     colorB=myColor, colorT=(255, 255, 255),
                                     colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        # Determine if we need to sound the alarm
        safety_alert = len(missing_items) > 0 or non_safety_detected
        
        # Sound buzzer if safety alert is triggered (without displaying warning message)
        current_time = time()
        if safety_alert and (current_time - last_beep_time) > beep_interval:
            winsound.Beep(beep_frequency, beep_duration)
            last_beep_time = current_time

        # Display frame
        cv2.imshow("PPE Compliance Monitoring", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting PPE compliance monitoring system...")
    detect_with_camera()