import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import time

# Load the trained model
model = load_model('best_model.h5')

# Posture labels
posture_labels = {
    0: "Correct post",
    1: "(Head down)",
    2: "(Turning)",
    3: "(Lean)",
    4: "(Head tilted)",
    5: "(Sloping shoulder)",
    6: "(Lying down)",
    7: "(Hands up)"
}

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

# Frame rate control
fps = 10  # Target: 10 predictions per second
interval = 1 / fps  # Time interval between predictions
last_prediction_time = time.time()

print("Press 'q' to quit the application.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB format (required by MediaPipe)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Convert back to BGR for displaying in OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Make predictions at the defined frame rate
        current_time = time.time()
        if current_time - last_prediction_time >= interval:
            # Extract pose landmarks
            landmarks = results.pose_landmarks.landmark
            x_coords = [lm.x for lm in landmarks]
            y_coords = [lm.y for lm in landmarks]

            # Combine x and y coordinates into a single feature vector
            features = np.array(x_coords + y_coords).reshape(1, -1)

            # Predict the posture
            predictions = model.predict(features)
            predicted_class = np.argmax(predictions)
            predicted_label = posture_labels[predicted_class]

            # Update the last prediction time
            last_prediction_time = current_time

            # Display the predicted class and confidence
            confidence = np.max(predictions) * 100
            cv2.putText(image, f"Predicted: {predicted_label} (Class {predicted_class})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        cv2.putText(image, "No pose detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Posture Prediction", image)

    # Check for user input to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("✅ Real-time inference ended successfully.")