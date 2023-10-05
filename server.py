import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import numpy as np
import time

'''
traffic signal switch algorithm
vechicle detection
vechicle counter

'''

# Create a Tkinter window
root = tk.Tk()
root.title("Video Streaming")

class Detection:
    
    def __init__(self):
        cfg_file = "yolov4-tiny.cfg"
        weights_file = "yolov4-tiny.weights"
        self.net = cv2.dnn.readNet(weights_file, cfg_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        classes_file = "coco.names"
        with open(classes_file, 'r') as f:
            self.classes = f.read().strip().split('\n')

        self.vechicle_count = 0
        self.object_dict = {}

    def detect(self,frame):
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

                if confidence > 0.3:
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
        self.object_dict = {}

        
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 0,255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Calculate the centroid of the detected object
            centroid_x = x + w // 2
            centroid_y = y + h // 2

            # Check if an existing object is close to this centroid
            matched_object = None
            for obj_id, (prev_centroid_x, prev_centroid_y) in self.object_dict.items():
                distance = np.sqrt((centroid_x - prev_centroid_x) ** 2 + (centroid_y - prev_centroid_y) ** 2)
                if distance < 20:  # Adjust this threshold as needed
                    matched_object = obj_id
                    break
            
            if matched_object is not None:
                # Update the centroid of the matched object
                self.object_dict[matched_object] = (centroid_x, centroid_y)
            else:
                # Add a new object to the dictionary
                self.object_dict[len(self.object_dict)] = (centroid_x, centroid_y)

        # Vehicle count is the number of unique objects in this frame
        vechicle_count = len(self.object_dict)

        return frame,vechicle_count


active_frame = 0  # Initialize to the first frame as active
max_count = 0
#signal_states = ["Green", "Red", "Red", "Red"]  # Each frame starts with Green
current_frame = 0
switch_interval = 8
signal_timers = [switch_interval] * 4 

#video_paused = [False, False, False, False]

signal_states = ["Red", "Red", "Red", "Red"]
video_paused = [True, True, True, True]


detection = Detection()


first_time = True
signal_durations = {
    "case1": {"green": 5},
    "case2": {"green": 20},
    "case3": {"green": 30},
    "case4": {"green": 60},
}

vechicle_count1,vechicle_count2,vechicle_count3,vechicle_count4 = 0,0,0,0


def count_condition(max_count):
    if max_count > 30:
        highest_priority_case = "case4"
    elif 11 <= max_count <= 30:
        highest_priority_case = "case3"
    elif 1 <= max_count <= 10:
        highest_priority_case = "case2"
    else:
        highest_priority_case = "case1"

    return highest_priority_case

max_count_used = False 
label_frame = {}
def switch_traffic_signal():
    global current_frame, active_frame, max_count,label_frame

    while True:
        current_frame = (current_frame + 1) % 4
        active_frame = current_frame
        signal_states[current_frame] = "Green"
        for i in range(4):
            if i != current_frame:
                signal_states[i] = "Red"
                video_paused[i] = True 
            else:
                video_paused[i] = False


        highest_priority_case = count_condition(max_count)
        duration = signal_durations[highest_priority_case]["green"]
        print("vechicle count : {} frame {} duration is {}".format(max_count,current_frame,duration))
        label_frame[current_frame] = duration

        if max_count > 0 and max_count >= duration:
            time.sleep(switch_interval + duration)
        else:
            time.sleep(switch_interval)


signal_thread = threading.Thread(target=switch_traffic_signal)
signal_thread.daemon = True
signal_thread.start()


def update_frame_counts(frame,count):
    frame_counts = {}
    frame_counts[frame] = count  
    return frame_counts

# Function to update the video frames
def update_frames():
    global first_time,current_frame,vechicle_count1,vechicle_count2,vechicle_count3,vechicle_count4,max_count,label_frame
    cap1 = cv2.VideoCapture('video.mp4')
    cap2 = cv2.VideoCapture('video2.mp4')
    cap3 = cv2.VideoCapture('video.mp4')
    cap4 = cv2.VideoCapture('video.mp4')
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        ret4, frame4 = cap4.read()

        if not ret1 or not ret2 or not ret3 or not ret4:
            break

        if signal_states[0] == "Red":
            cap1.set(cv2.CAP_PROP_POS_FRAMES, cap1.get(cv2.CAP_PROP_POS_FRAMES) - 1)  
        if signal_states[1] == "Red":
            cap2.set(cv2.CAP_PROP_POS_FRAMES, cap2.get(cv2.CAP_PROP_POS_FRAMES) - 1)  
           
        if signal_states[2] == "Red":
            cap3.set(cv2.CAP_PROP_POS_FRAMES, cap3.get(cv2.CAP_PROP_POS_FRAMES) - 1)  
            
        if signal_states[3] == "Red":
            cap4.set(cv2.CAP_PROP_POS_FRAMES, cap4.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            
        
        # Resize frames to fit the window
        frame1 = cv2.resize(frame1, (640, 400))
        frame2 = cv2.resize(frame2, (640, 400))
        frame3 = cv2.resize(frame3, (640, 400))
        frame4 = cv2.resize(frame4, (640, 400))


   

        max_count = max(vechicle_count1,vechicle_count2,vechicle_count3,vechicle_count4)                 

        if current_frame != active_frame:
            vechicle_count1, vechicle_count2, vechicle_count3, vechicle_count4 = 0, 0, 0, 0  # No vehicle count available
        else:
            frame1, vechicle_count1 = detection.detect(frame1) if current_frame != 0 else (frame1, 0)
            frame2, vechicle_count2 = detection.detect(frame2) if current_frame != 1 else (frame2, 0)
            frame3, vechicle_count3 = detection.detect(frame3) if current_frame != 2 else (frame3, 0)
            frame4, vechicle_count4 = detection.detect(frame4) if current_frame != 3 else (frame4, 0)




        if vechicle_count1 != 0:
            
            label1_signal.config(text=f"Traffic Signal: {signal_states[0]}, Vechicle Count: {vechicle_count1}")
        else:
            label1_signal.config(text=f"Traffic Signal: {signal_states[0]}")
        
        if vechicle_count2 != 0:
            result = label_frame[1] if label_frame[1] else "NIL"
            label2_signal.config(text=f"Traffic Signal: {signal_states[1]}, Vechicle Count: {vechicle_count2}")
        else:
            label2_signal.config(text=f"Traffic Signal: {signal_states[1]}")

        if vechicle_count3 !=0:
            label3_signal.config(text=f"Traffic Signal: {signal_states[2]}, Vechicle Count: {vechicle_count3}")
        else:
            label3_signal.config(text=f"Traffic Signal: {signal_states[2]}")

        if vechicle_count4 !=0:
            label4_signal.config(text=f"Traffic Signal: {signal_states[3]}, Vechicle Count: {vechicle_count4}")
        else:
            label4_signal.config(text=f"Traffic Signal: {signal_states[3]}")



        # Convert frames to Tkinter format
        img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        img3 = Image.fromarray(cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB))
        img4 = Image.fromarray(cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB))

        tk_img1 = ImageTk.PhotoImage(image=img1)
        tk_img2 = ImageTk.PhotoImage(image=img2)
        tk_img3 = ImageTk.PhotoImage(image=img3)
        tk_img4 = ImageTk.PhotoImage(image=img4)

        label1.config(image=tk_img1)
        label2.config(image=tk_img2)
        label3.config(image=tk_img3)
        label4.config(image=tk_img4)
        

        label1.image = tk_img1
        label2.image = tk_img2
        label3.image = tk_img3
        label4.image = tk_img4

    # Release video capture objects
    cap1.release()
    cap2.release()
    cap3.release()
    cap4.release()

# Create labels for displaying video frames
label1 = ttk.Label(root)
label2 = ttk.Label(root)
label3 = ttk.Label(root)
label4 = ttk.Label(root)

label1.grid(row=0, column=0)
label2.grid(row=0, column=1)
label3.grid(row=2, column=0)
label4.grid(row=2, column=1)

# Create labels for displaying traffic signals and timers below each frame
label1_signal = ttk.Label(root, text="", font=("Helvetica", 21))
label2_signal = ttk.Label(root, text="", font=("Helvetica", 21))
label3_signal = ttk.Label(root, text="", font=("Helvetica", 21))
label4_signal = ttk.Label(root, text="", font=("Helvetica", 21))


label1_signal.grid(row=1, column=0, padx=10)
label2_signal.grid(row=1, column=1, padx=10)
label3_signal.grid(row=3, column=0, padx=10)
label4_signal.grid(row=3, column=1, padx=10)


# Create a thread for updating frames
update_thread = threading.Thread(target=update_frames)
update_thread.daemon = True
update_thread.start()

root.mainloop()