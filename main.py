from matplotlib import pyplot as plt

from src.plate_character_detector import PlateCharacterDetector
from src.utils import show_plots, create_image_dict


plate_character_detector = PlateCharacterDetector()

plate_character_detector.load_image(
    file_path="./data/example/indonesia_nopol_5.JPG")

image_bgr = plate_character_detector.image_bgr
image_gray = plate_character_detector.image_gray
image_blur = plate_character_detector.apply_gaussian_blur(
    image_gray=image_gray, kernel_size=(7, 7))
image_bw = plate_character_detector.apply_thresholding(image_gray=image_blur)
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
