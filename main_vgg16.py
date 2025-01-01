from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Charger le classificateur pour détecter les visages
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#classifier = load_model('vgg16_model.h5')
classifier = load_model('vgg16_model.h5')

# Étiquettes pour les émotions et leurs poids (EW)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_weights = {
    'Angry': 0.25,
    'Disgust': 0,
    'Fear': 0.3,
    'Happy': 0.6,
    'Neutral': 0.9,
    'Sad': 0.3,
    'Surprise': 0.5
}

# Catégorisation de la concentration
def classify_concentration(ci):
    if ci >= 0.7:
        return "Highly Concentrated"
    elif 0.4 <= ci < 0.7:
        return "Nominally Concentrated"
    else:
        return "Not Concentrated"

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            # Convertir l'image en niveaux de gris en une image RGB
            roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
            roi = roi_rgb.astype('float32') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Prédire l'émotion
            prediction = classifier.predict(roi)
            probabilities = prediction[0]  # Probabilités pour chaque émotion
            dominant_emotion_index = np.argmax(probabilities)
            dominant_emotion = emotion_labels[dominant_emotion_index]

            # Calculer l'indice de concentration (CI)
            dominant_emotion_weight = emotion_weights[dominant_emotion]
            ci = probabilities[dominant_emotion_index] * dominant_emotion_weight
            concentration_level = classify_concentration(ci)

            # Afficher les résultats sur l'écran
            label_position = (x, y - 10)
            cv2.putText(frame, f"{dominant_emotion} ({ci:.2f})", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"{concentration_level}/{ci*100:.2f}%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher le flux vidéo
    cv2.imshow('Concentration Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
