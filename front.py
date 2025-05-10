import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import joblib
import csv
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignSpeak - AI Sign Language Translator")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")

        # Create main frames
        self.video_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.video_frame.pack(fill="both", expand=True)
        
        self.control_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.control_frame.pack(fill="x", side="bottom", pady=10)

        # Video display
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(pady=10)

        # Prediction display with larger font and contrast
        self.predict_label = tk.Label(self.video_frame, text="Prediction: Waiting...", 
                                      font=("Arial", 32, "bold"), bg="#1e1e1e", fg="#4CAF50")
        self.predict_label.pack(pady=20)

        # Load the model
        try:
            self.model = joblib.load("gesture_model.pkl")
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            self.model = None
            print("⚠️ gesture_model.pkl not found. Run the training script after recording gestures.")

        # Control buttons
        btn_frame = tk.Frame(self.control_frame, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="Start", command=self.start_camera, 
                                   bg="#4CAF50", fg="white", font=("Arial", 14))
        self.start_btn.pack(side="left", padx=10)

        self.pause_btn = tk.Button(btn_frame, text="Pause", command=self.pause_camera, 
                                   bg="#FFC107", fg="black", font=("Arial", 14))
        self.pause_btn.pack(side="left", padx=10)

        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_camera, 
                                  bg="#F44336", fg="white", font=("Arial", 14))
        self.stop_btn.pack(side="left", padx=10)

        record_frame = tk.Frame(btn_frame, bg="#1e1e1e")
        record_frame.pack(side="left", padx=10)
        
        self.record_btn = tk.Button(record_frame, text="Record Sample", command=self.record_sample, 
                                   bg="#2196F3", fg="white", font=("Arial", 14))
        self.record_btn.pack(side="top", pady=5)
        
        label_frame = tk.Frame(record_frame, bg="#1e1e1e")
        label_frame.pack(side="top")
        
        tk.Label(label_frame, text="Gesture Label:", bg="#1e1e1e", fg="white").pack(side="left")
        
        self.current_label = tk.StringVar(value="OK")  # default class
        self.label_entry = tk.Entry(label_frame, textvariable=self.current_label, 
                                   width=10, font=("Arial", 14))
        self.label_entry.pack(side="left", padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#333333", fg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Camera status flags
        self.camera_on = False
        self.paused = False
        self.capture = None

        # Initialize MediaPipe hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for better performance
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=15)  # Store recent predictions
        self.last_stable_prediction = "None"
        self.prediction_counter = {}  # Count occurrences of each prediction
        
        # Confidence threshold for predictions
        self.confidence_threshold = 0.0  # Default value, can be adjusted

    def start_camera(self):
        if not self.camera_on:
            self.capture = cv2.VideoCapture(0)
            self.camera_on = True
            self.paused = False
            self.update_frame()
            self.status_var.set("Camera active")
        elif self.camera_on and self.paused:
            self.paused = False
            self.status_var.set("Camera unpaused")

    def pause_camera(self):
        if self.camera_on:
            self.paused = not self.paused
            status = "Camera paused" if self.paused else "Camera active"
            self.status_var.set(status)

    def stop_camera(self):
        self.camera_on = False
        self.paused = False
        if self.capture and self.capture.isOpened():
            self.capture.release()
        self.capture = None
        self.video_label.configure(image='')
        self.predict_label.config(text="Prediction: Stopped")
        self.status_var.set("Camera stopped")

    def record_sample(self):
        if not self.capture or not self.capture.isOpened():
            self.status_var.set("Error: Camera not available")
            return
            
        ret, frame = self.capture.read()
        if not ret:
            self.status_var.set("Error: Failed to capture frame")
            return
            
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img)
        
        if not results.multi_hand_landmarks:
            self.status_var.set("No hand detected")
            return
            
        landmarks = results.multi_hand_landmarks[0].landmark
        flat = [coord for lm in landmarks for coord in (lm.x, lm.y)]
        label = self.current_label.get()
        
        with open("gesture_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + flat)
            
        self.status_var.set(f"Recorded sample for '{label}'")
        print(f"✅ Recorded sample for '{label}'")

    def update_frame(self):
        if self.camera_on and not self.paused:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror effect
                
                # Process hand landmarks
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                # Create a copy for visualization
                viz_frame = frame.copy()
                
                if results.multi_hand_landmarks:
                    # Draw landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            viz_frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Extract landmarks for prediction
                        landmark_list = []
                        for lm in hand_landmarks.landmark:
                            landmark_list.append((lm.x, lm.y))
                            
                        # Get prediction and update history
                        current_prediction = self.recognize_gesture(landmark_list)
                        
                        # Update GUI with smoothed prediction
                        self.update_prediction_display(current_prediction)
                else:
                    # No hand detected - clear prediction after a delay
                    self.prediction_history.append("No hand")
                    if len(self.prediction_history) >= 10 and all(p == "No hand" for p in list(self.prediction_history)[-5:]):
                        self.predict_label.config(text="Prediction: Waiting for hand...")
                
                # Convert to Tkinter format
                img_pil = Image.fromarray(cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img_pil)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        if self.camera_on:
            self.root.after(10, self.update_frame)

    def recognize_gesture(self, landmarks):
        flat = [coord for (x, y) in landmarks for coord in (x, y)]
        if len(flat) != 42:  # Check for proper dimensions
            return "Invalid"
            
        if self.model is None:
            return "Model not loaded"
            
        try:
            # Get raw prediction from model
            prediction = self.model.predict([flat])[0]
            
            # Add prediction to history
            self.prediction_history.append(prediction)
            
            return prediction
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "Error"

    def update_prediction_display(self, current_prediction):
        # Update prediction counts
        self.prediction_counter = {}
        for pred in self.prediction_history:
            if pred not in self.prediction_counter:
                self.prediction_counter[pred] = 0
            self.prediction_counter[pred] += 1
        
        # Find the most common prediction
        most_common = None
        max_count = 0
        for pred, count in self.prediction_counter.items():
            if count > max_count and pred not in ["Invalid", "Error", "No hand"]:
                max_count = count
                most_common = pred
        
        # Only update display if we have a stable prediction
        # (appears in more than 60% of recent frames)
        if most_common and max_count >= 0.6 * len(self.prediction_history):
            if most_common != self.last_stable_prediction:
                self.last_stable_prediction = most_common
                self.predict_label.config(text=f"Prediction: {most_common}")
        
    def __del__(self):
        if hasattr(self, 'capture') and self.capture and self.capture.isOpened():
            self.capture.release()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()