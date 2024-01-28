import cv2
import numpy as np
import base64
from ultralytics import YOLO
import threading
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse

# Create your views here.
def home(request):
    return render(request, 'home.html')

class VideoStream:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]
        self.is_streaming = False
        self.lock = threading.Lock()

    def start_stream(self):
        self.is_streaming = True
        threading.Thread(target=self.update_frame, args=()).start()

    def stop_stream(self):
        self.is_streaming = False

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def update_frame(self):
        while self.is_streaming:
            success, frame = self.video_capture.read()
            with self.lock:
                self.current_frame = frame

def security(request):
    if request.method == 'GET':
        return render(request, 'security.html')

    stream = request.session.get('stream')
    if stream is None:
        stream = VideoStream()
        request.session['stream'] = stream

    if request.POST.get('action') == 'start':
        stream.start_stream()
        return StreamingHttpResponse(stream_generator(stream), content_type='multipart/x-mixed-replace; boundary=frame')

    if request.POST.get('action') == 'stop':
        stream.stop_stream()
        return StreamingHttpResponse()

def stream_generator(stream):
    model = YOLO("./models/best_lentes.pt")

    while True:
        frame = stream.get_frame()

        # Realizar la detección de objetos en cada fotograma
        resultados = model.predict(frame, imgsz=640, conf=0.50)
        annotated_image = resultados[0].plot()

        # Convertir la imagen anotada a base64 para enviarla en la respuesta
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_base64.encode('utf-8') + b'\r\n')

        # Liberar recursos
        cv2.waitKey(1)
        cv2.destroyAllWindows()