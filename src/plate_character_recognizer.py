from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2


class PlateCharacterRecognizer:

    def __init__(self):
        self.model = None
        self.lb = LabelEncoder()

    def predict(self, character_image):
        preprocessed_character = self._preprocessing(
            character_image=character_image)
        preprocessed_character = preprocessed_character[np.newaxis, :]
        
        predicted_confidence = np.max(self.model.predict(preprocessed_character))
        predicted_character = self.lb.inverse_transform([np.argmax(self.model.predict(preprocessed_character))])[0]
        
        return predicted_character, predicted_confidence

    def _preprocessing(self, character_image):
        size = (80, 80)
        image = cv2.resize(character_image, size)
        return np.stack((image, ) * 3, axis=-1)

    def load_model(self, file_path='./src/models/MobileNetV2_plate_character_recognizer.json'):
        json_file = open(file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        print("[INFO] Model architecture loaded successfully...")

    def load_weights(self, file_path='./src/models/plate_character_classifier_weight_01.h5'):
        if self.model is None:
            raise Exception("Please load the model before loading the weights")

        self.model.load_weights(file_path)
        print("[INFO] Model weights loaded successfully...")

    def load_classes_label(self, file_path='./src/models/plate_character_classes.npy'):
        self.lb.classes_ = np.load(file_path)
        print("[INFO] Classes label loaded successfully...")
