from flask import Flask, request
from flask.templating import render_template
import numpy as np
import cv2
import base64

from src.plate_character_detector import PlateCharacterDetector
from src.plate_character_recognizer import PlateCharacterRecognizer
from src.utils import save_plots, create_image_dict

app = Flask(__name__)

plate_character_detector = PlateCharacterDetector()
plate_character_recognizer = PlateCharacterRecognizer()


def process_license_plate(image):
    plate_character_detector.load_image(image=image)
    (character_rois, crop_characters) = plate_character_detector.detect_characters()
    plate_character_recognizer.load_model()
    plate_character_recognizer.load_weights()
    plate_character_recognizer.load_classes_label()

    characters_image = []
    characters = ""

    for character in crop_characters:
        predicted_character, confidence_rate = plate_character_recognizer.predict(
            character)
        characters += predicted_character
        character_label = "{}, {}%".format(
            predicted_character, round(confidence_rate * 100, 2))
        characters_image.append(create_image_dict(
            character, character_label, cmap="gray"))

    crop_characters_plot = save_plots((9, 4), ncols=len(characters_image), nrows=1,
                                      images=characters_image, font_size=9)

    return crop_characters_plot, characters


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_bytes = request.files['license-photo'].read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)

        crop_characters_plot, characters = process_license_plate(image)

        _, buffer = cv2.imencode('.jpg', img=crop_characters_plot)
        base64_crop_characters_plot = base64.b64encode(buffer).decode("UTF-8")

        _, buffer = cv2.imencode('.jpg', img=image)
        raw_image = base64.b64encode(buffer).decode("UTF-8")

        return render_template('result.html', raw_image=raw_image, preprocessed_image=base64_crop_characters_plot,
                               classified_text=characters)

    return render_template('index.html', raw_image=None, preprocessed_image=None, classified_text=None)
