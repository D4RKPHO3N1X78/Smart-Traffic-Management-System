import os
import cv2
import numpy as np

# Global model instance initialized lazily
_model = None

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

def get_yolo_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            # Load YOLOv8 nano model (downloads automatically if not cached)
            _model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"[Warning] Failed to load YOLO model: {e}")
            _model = False
    return _model

def determine_traffic_status(vehicle_count: int) -> str:
    """
    Categorize traffic density based on detected vehicle count.
    """
    if vehicle_count <= 5:
        return "Clear"
    elif vehicle_count <= 15:
        return "Moderate"
    else:
        return "Heavy"

def process_image(file_path: str, output_path: str):
    model = get_yolo_model()
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image file at {file_path}")

    vehicle_count = 0

    if model:
        results = model(file_path)
        result = results[0]
        
        # Filter for vehicle classes (car, motorcycle, bus, truck)
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id in VEHICLE_CLASSES and conf >= 0.25:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 128), 2)
                # Draw text background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 128), -1)
                cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        # Fallback contour detection if model is unavailable
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 128), 2)
                vehicle_count += 1

    # Overlay overall vehicle count banner
    status = determine_traffic_status(vehicle_count)
    h, w, _ = image.shape
    banner_height = max(40, int(h * 0.08))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    cv2.putText(
        image,
        f"Vehicles Detected: {vehicle_count} | Status: {status}",
        (15, int(banner_height * 0.65)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imwrite(output_path, image)
    return vehicle_count

def process_video(file_path: str, output_path: str):
    model = get_yolo_model()
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file at {file_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    max_vehicle_count = 0
    frame_idx = 0
    sample_rate = 2  # Process every 2nd frame to optimize processing time

    last_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_count = 0

        if frame_idx % sample_rate == 0 or frame_idx == 1:
            last_boxes = []
            if model:
                results = model(frame, verbose=False)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if cls_id in VEHICLE_CLASSES and conf >= 0.25:
                        current_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
                        last_boxes.append((x1, y1, x2, y2, label))
            else:
                last_boxes = []
        else:
            current_count = len(last_boxes)

        max_vehicle_count = max(max_vehicle_count, current_count)

        # Draw stored boxes
        for (x1, y1, x2, y2, label) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 128), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Banner overlay
        status = determine_traffic_status(max_vehicle_count)
        banner_height = max(40, int(height * 0.08))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(
            frame,
            f"Peak Vehicles: {max_vehicle_count} | Frame Count: {current_count} | Status: {status}",
            (15, int(banner_height * 0.65)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        out.write(frame)

    cap.release()
    out.release()
    return max_vehicle_count

def process_media(file_path: str, is_video: bool = False):
    os.makedirs("static", exist_ok=True)
    basename = os.path.basename(file_path)
    output_filename = f"processed_{basename}"
    output_path = os.path.join("static", output_filename)

    if is_video:
        # Ensure mp4 extension for output
        if not output_filename.endswith(".mp4"):
            output_filename += ".mp4"
            output_path = os.path.join("static", output_filename)
        vehicle_count = process_video(file_path, output_path)
    else:
        vehicle_count = process_image(file_path, output_path)

    return vehicle_count, output_path
