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
    images = [create_image_dict(image_bgr, "Color", None), create_image_dict(image_gray, "Gray", "gray")]
    show_plots((6, 2), ncols=2, nrows=1, images=images)
