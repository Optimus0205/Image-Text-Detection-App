import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

# Load the custom model once at import time (much faster than reloading on every request)
model_loaded = YOLO('best.pt')


def predict(image):
    # Read the image from the uploaded file
    image_bytes = np.frombuffer(image.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Perform inference
    results = model_loaded(image, conf=0.50)[0]

    # Create Detections with labels
    detections = sv.Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )

    # Get class names from the model
    class_names = model_loaded.names

    # Annotate boxes and labels with clean names
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,     # Adjust text size if needed
        text_thickness=1,
    )

    # Annotate the image with detections and their labels
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=[f"{class_names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
    )

    # Encode annotated image to JPEG bytes (needs .tobytes() for io.BytesIO downstream)
    _, buffer = cv2.imencode('.jpg', annotated_image)
    image_bytes_out = buffer.tobytes()

    # Build JSON-serializable detection details (numpy/float32 types are not JSON serializable)
    detection_results = []
    for box, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
        detection_results.append({
            'class': class_names[int(cls)],
            'confidence': round(float(conf), 2),
            'box': [round(float(coord), 1) for coord in box]
        })

    return image_bytes_out, detection_results