import unittest
from matplotlib import pyplot as plt

from src.plate_character_detector import PlateCharacterDetector
from src.utils import show_plots, create_image_dict


class TestPlateCharacterDetector(unittest.TestCase):

    def setUp(self):
        self.plate_character_detector = PlateCharacterDetector()

    def test_load_image(self):
        file_path = "./data/example/indonesia_nopol_1.JPG"
        self.plate_character_detector.load_image(file_path)
        image_bgr = self.plate_character_detector.image_bgr
        image_gray = self.plate_character_detector.image_gray
        images = [create_image_dict(image_bgr, "Color", None), create_image_dict(
            image_gray, "Gray", "gray")]
        show_plots((6, 2), ncols=2, nrows=1, images=images)

    def test_detect_character(self):
        file_path = "./data/example/indonesia_nopol_5.JPG"
        self.plate_character_detector.load_image(file_path)
        (character_rois, crop_characters) = self.plate_character_detector.detect_characters()
        plt.figure(figsize=(5, 3))
        plt.axis(False)
        plt.imshow(character_rois)

        images = [create_image_dict(char, "", cmap="gray")
                  for char in crop_characters]
        show_plots((len(crop_characters), 2), ncols=len(
            crop_characters), nrows=1, images=images)
