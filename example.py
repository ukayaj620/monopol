from matplotlib import pyplot as plt
import numpy as np

from src.plate_character_recognizer import PlateCharacterRecognizer
from src.plate_character_detector import PlateCharacterDetector
from src.utils import show_plots, create_image_dict


plate_character_detector = PlateCharacterDetector()
plate_character_recognizer = PlateCharacterRecognizer()

plate_character_detector.load_image(
    file_path="./data/example/indonesia_nopol_4.JPG")

image_bgr = plate_character_detector.image_bgr
image_gray = plate_character_detector.image_gray
image_blur = plate_character_detector.apply_gaussian_blur(
    image_gray=image_gray, kernel_size=(7, 7))
image_bw = plate_character_detector.apply_thresholding(image_gray=image_gray)
image_erode = plate_character_detector.apply_erosion(
    image_bw=image_bw, kernel=plate_character_detector.create_kernel(3))

images = [
    create_image_dict(image_bgr, "Color", None),
    create_image_dict(image_gray, "Grayscaling", "gray"),
    create_image_dict(image_blur, "Bluring", "gray"),
    create_image_dict(image_bw, "Thresholding", "gray"),
    create_image_dict(image_erode, "Erosion", "gray")
]
show_plots((9, 4), ncols=3, nrows=2, images=images)

(character_rois, crop_characters) = plate_character_detector.detect_characters()

plt.figure(figsize=(5, 3))
plt.axis(False)
plt.imshow(character_rois)

images = [create_image_dict(char, "", cmap="gray")
          for char in crop_characters]
show_plots((len(crop_characters), 2), ncols=len(
    crop_characters), nrows=1, images=images)


plate_character_recognizer.load_model()
plate_character_recognizer.load_weights()
plate_character_recognizer.load_classes_label()

characters_image = []

for character in crop_characters:
    predicted_character, confidence_rate = plate_character_recognizer.predict(
        character)
    character_label = "{}, {}%".format(
        predicted_character, round(confidence_rate * 100, 2))
    characters_image.append(create_image_dict(
        character, character_label, cmap="gray"))

show_plots((9, 4), ncols=len(characters_image), nrows=1,
           images=characters_image, font_size=9)
