import cv2 # Import OpenCV for video capture
import tkinter as tk # Import Tkinter for GUI
from PIL import Image, ImageTk # Import Python Imaging Library for image processing
import mediapipe as mp # Import MediaPipe for hand tracking


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp: # Define the main application class
    def __init__(self, root): # Initialize the GUI window
        self.root = root # Create a root window
        self.root.title("SignSpeak - AI Sign Language Translator") # Set the window title
        self.root.geometry("800x600") # Set the window size
        self.root.configure(bg="#1e1e1e") # Set the background color

        self.video_label = tk.Label(self.root) # Create a label for video display
        self.video_label.pack() # Pack the label into the window

        self.predict_label = tk.Label(self.root, text="Prediction: ", font=("Arial", 24), bg="#1e1e1e", fg="white") # Create a label for displaying predictions
        self.predict_label.pack(pady=20) # Pack the prediction label

        btn_frame = tk.Frame(self.root, bg="#1e1e1e") # Create a frame for buttons
        btn_frame.pack(pady=10) # Pack the button frame

        self.start_btn = tk.Button(btn_frame, text="Start", command=self.start_camera, bg="#4CAF50", fg="white", font=("Arial", 14)) # Start button
        self.start_btn.pack(side="left", padx=10) # Pack start button

        self.pause_btn = tk.Button(btn_frame, text="Pause", command=self.pause_camera, bg="#FFC107", fg="black", font=("Arial", 14)) # Pause button
        self.pause_btn.pack(side="left", padx=10) # Pack pause button

        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_camera, bg="#F44336", fg="white", font=("Arial", 14)) # Stop button
        self.stop_btn.pack(side="left", padx=10) # Pack stop button

        # Camera status flags
        self.camera_on = False # Flag to check if camera is on
        self.paused = False # Flag to check if camera is paused
        self.capture = None # Video capture object

        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    def start_camera(self): # Method to start the camera
        if not self.camera_on: # If camera is not already on
            self.capture = cv2.VideoCapture(0) # Start video capture from the webcam
            self.camera_on = True # Set camera on flag
            self.paused = False # Reset paused state
            self.update_frame() # Start updating frames
        elif self.camera_on and self.paused: # If camera is paused
            self.paused = False # Just resume frame updates

    def pause_camera(self): # Method to pause/resume the camera
        if self.camera_on: # Only toggle if camera is on
            self.paused = not self.paused # Toggle pause state

    def stop_camera(self): # Method to stop the camera
        self.camera_on = False # Set camera off flag
        self.paused = False # Reset paused state
        if self.capture and self.capture.isOpened(): # If capture object is valid
            self.capture.release() # Release the video capture
        self.capture = None # Clear the capture object
        self.video_label.configure(image='') # Clear video feed from label

    def update_frame(self): # Method to update the video frame
        if self.camera_on and not self.paused: # Only update if camera is on and not paused
            ret, frame = self.capture.read() # Read a frame from the webcam
            if ret: # If a frame is successfully captured
                frame = cv2.flip(frame, 1) # Flip the frame horizontally for a mirror effect
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame from BGR to RGB format

                results = self.hands.process(img) # Process the frame with MediaPipe Hands
                if results.multi_hand_landmarks: # If hands are detected
                    for landmarks in results.multi_hand_landmarks: # Loop through detected hands
                        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS) # Draw landmarks on the frame
                        landmark_list = [] # List to store landmark coordinates
                        for lm in landmarks.landmark: # Loop through each landmark
                            landmark_list.append((lm.x, lm.y)) # Append x, y coordinates to the list
                        prediction = self.recognize_gesture(landmark_list) # Call the gesture recognition function
                        self.predict_label.config(text=f"Prediction: {prediction}")  # Update the prediction label

                img = Image.fromarray(img) # Convert the frame to a PIL image
                imgtk = ImageTk.PhotoImage(image=img)  # Convert the PIL image to a PhotoImage for Tkinter

                self.video_label.imgtk = imgtk # Keep a reference to the PhotoImage object
                self.video_label.configure(image=imgtk) # Update the label with the new image

        if self.camera_on: # If camera is still on, keep looping
            self.root.after(10, self.update_frame) # Schedule the next frame update

    def recognize_gesture(self, landmarks): # Gesture recognition function
        thumb_tip = landmarks[4] # Thumb tip landmark
        index_tip = landmarks[8] # Index finger tip landmark
        distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5 # Calculate distance between thumb and index finger tips
        if distance < 0.1: # If distance is less than a threshold, recognize as "OK" gesture
            return "OK" 
        else: # If distance is greater, recognize as "Not OK" gesture
            return "Not Recognized"

    def __del__(self): # Destructor to release the video capture when the object is deleted
        if self.capture and self.capture.isOpened(): # Check if the capture is opened
            self.capture.release() # Release the video capture


# Run the app
root = tk.Tk() # Create the main window
app = SignLanguageApp(root) # Create an instance of the SignLanguageApp class
root.mainloop() # Start the Tkinter main loop to run the application
