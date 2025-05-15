import cv2
import numpy as np
import keras
import mediapipe as mp

model = keras.models.load_model("america_sign_language_2.keras")
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


while True:
  ret, frame = cap.read()
  if not ret:
    break

  image_rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  result = hands.process(image_rgb)

  if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
      h, w, c = frame.shape
      landmark_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
      x_min = int(np.min(landmark_array[:, 0]))
      y_min = int(np.min(landmark_array[:, 1]))
      x_max = int(np.max(landmark_array[:, 0]))
      y_max = int(np.max(landmark_array[:, 1]))

      padding = 20
      x_min = max(x_min - padding, 0)
      y_min = max(y_min - padding, 0)
      x_max = max(x_max + padding, w)
      y_max = min(y_max + padding, h)

      roi = frame[y_min:y_max, x_min:x_max]
      if roi.size == 0:
        continue
      
      roi_resized = cv2.resize(roi, (224, 224))
      prediction = model.predict(np.expand_dims(roi_resized, axis=0))[0]
      label_idx = np.argmax(prediction)
      confidence = prediction[label_idx]
      
      label_text = f"{labels[label_idx]}: {confidence:.2f}"
      cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
      cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

      mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


  cv2.imshow('Real-Time Hand Sign Detection', frame)
  if cv2.waitKey(1) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()