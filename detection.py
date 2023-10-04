import cv2
import numpy as np

class Detection:
    def __init__(self):
        cfg_file = "yolov4-tiny.cfg"
        weights_file = "yolov4-tiny.weights"
        self.net = cv2.dnn.readNet(weights_file, cfg_file)

        classes_file = "coco.names"
        with open(classes_file, 'r') as f:
            self.classes = f.read().strip().split('\n')


    def detect(self,frame,conf):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()

        detections = self.net.forward(layer_names)
        
        class_ids = []
        confidences = []
        boxes = []

        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

        for i in range(len(boxes)):
            for i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame


def worker(frame_que,result_que):
    detect = Detection()
    
    while True:
        frame = frame_que.get()

        if frame is None:
            break
    
        result = detect.detect(frame,0.5)
        result_que.put(result)

video_file = "video.mp4"
cap = cv2.VideoCapture(video_file)

detect = Detection()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detect.detect(frame,0.5)

    cv2.imshow("YOLOv4 Tiny Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
