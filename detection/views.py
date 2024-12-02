from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from PIL import Image
from ultralytics import YOLO
import cv2

# Load YOLOv8 model globally
model = YOLO('yolov8s.pt')  # Load the YOLOv8 small model

# Homepage
def home(request):
    return render(request, 'base.html')

# Image Upload for Detection
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        try:
            # Open the uploaded image
            image = Image.open(image_file)
            
            # Perform object detection using YOLOv8
            results = model(image)
            
            # Parse detections into a structured format
            detections = []
            for r in results[0].boxes:
                detections.append({
                    "x1": int(r.xyxy[0][0]),
                    "y1": int(r.xyxy[0][1]),
                    "x2": int(r.xyxy[0][2]),
                    "y2": int(r.xyxy[0][3]),
                    "confidence": float(r.conf[0]),
                    "class": int(r.cls[0]),
                    "name": model.names[int(r.cls[0])]
                })
            
            return JsonResponse({'detections': detections})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return render(request, 'upload_image.html')

# Live Video Stream with YOLOv8 Detection
def stream_video(request):
    cap = cv2.VideoCapture(0)

    def generate_frames():
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                # Perform object detection on the frame
                results = model(frame)
                
                for r in results[0].boxes:
                    x1, y1, x2, y2 = map(int, r.xyxy[0])  # Bounding box coordinates
                    conf = float(r.conf[0])  # Confidence score
                    cls = int(r.cls[0])  # Class index
                    label = model.names[cls]  # Class name
                    
                    # Draw bounding boxes and labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            cap.release()

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
