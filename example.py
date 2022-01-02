from matplotlib import pyplot as plt

from src.plate_character_recognizer import PlateCharacterRecognizer
from src.plate_character_detector import PlateCharacterDetector
from src.utils import show_plots, create_image_dict

"""
Instantiate Object from 
Class PlateCharacterDetector and PlateCharacterRecognizer
"""
plate_character_detector = PlateCharacterDetector()
plate_character_recognizer = PlateCharacterRecognizer()

"""
Load image by providing file_path.

load_image() method require at least one parameters:
1. file_path: file path of an image in string format
2. image_array: numpy array representation of an image
"""
plate_character_detector.load_image(
    file_path="./data/example/indonesia_nopol_3.JPG")

"""
BGR representation of loaded image
"""
image_bgr = plate_character_detector.image_bgr

"""
Grayscale representation of loaded image
"""
image_gray = plate_character_detector.image_gray

"""
Gaussian Blur representation of loaded image.

apply_gaussian_blur() method required two parameters:
1. image_gray: Grayscaled image
2. kernel_size: The size of kernel in tuple format
"""
image_blur = plate_character_detector.apply_gaussian_blur(
    image_gray=image_gray, kernel_size=(7, 7))

"""
Thresholding representation of loaded image.

apply_thresholding() method required two parameters:
1. image_gray: Grayscaled image
2. threshold_value: An byte represent thresholding bounding value (0 - 255)
"""
image_bw = plate_character_detector.apply_thresholding(
    image_gray=image_gray, threshold_value=180)

"""
Erosion Morphology representation of loaded image.

apply_erosion() method required two parameters:
1. image_bw: Black and White image (after thresholding)
2. kernel: Structuring element kernel for morphology
"""
image_erode = plate_character_detector.apply_erosion(
    image_bw=image_bw, kernel=plate_character_detector.create_kernel(3))


"""
Show the processed image representation using matplotlib.

create_image_dict() method required three parameter;
1. image: numpy array representation of an image
2. label: the image label in string
3. cmap(optional): matplotlib color map representation

show_plots() method required four parameters:
1. figure_size: The size of the pyplot figure in tuple format (width, height)
2. ncols: Number of column showed in the figure
3. nrows: Number of row showed in the figure
4. images: Array of image_dictionary created using create_image_dict method
"""
images = [
    create_image_dict(image_bgr, "Color"),
    create_image_dict(image_gray, "Grayscaling", "gray"),
    create_image_dict(image_blur, "Bluring", "gray"),
    create_image_dict(image_bw, "Thresholding", "gray"),
    create_image_dict(image_erode, "Erosion", "gray")
]
show_plots((9, 4), ncols=3, nrows=2, images=images)

"""
Detect character after applying image preprocessing using grayscaling, bluring, thresholding, and morphology

detect_character() method doesn't required any parameter.
method returns:
1. character_rois: detected character from a plate image 
                   (represented as whole plate image with bounding boxes of the detected character)
2. crop_characters: array of cropped version of detected characters, each cropped characters
                    represented in numpy array form.
"""
(character_rois, crop_characters) = plate_character_detector.detect_characters()

"""
Visualize the character rois
"""
plt.figure(figsize=(5, 3))
plt.axis(False)
plt.imshow(character_rois)

"""
Visualize the cropped characters
"""
images = [create_image_dict(char, "", cmap="gray")
          for char in crop_characters]
show_plots((len(crop_characters), 2), ncols=len(
    crop_characters), nrows=1, images=images)


"""
Load the plate character recognizer deep learning model

In order to use the model, respect the several steps below:
1. Loading the model structure using load_model() method.
2. Loading the model weights using load_weights() method.
3. Loading the class categories label of the classification 
   layer of the model using load_classes_label() method.
"""
plate_character_recognizer.load_model()
plate_character_recognizer.load_weights()
plate_character_recognizer.load_classes_label()

characters_image = []

"""
Recognize cropped characters using the deep learning model
using predict() method.

predict() method require one parameter:
character_image: numpy array representation of a single character image.
"""

for character in crop_characters:
    predicted_character, confidence_rate = plate_character_recognizer.predict(
        character)
    character_label = "{}, {}%".format(
        predicted_character, round(confidence_rate * 100, 2))
    characters_image.append(create_image_dict(
        character, character_label, cmap="gray"))

"""
Visualize the predicted or recognized cropped plate character result
"""
show_plots((9, 4), ncols=len(characters_image), nrows=1,
           images=characters_image, font_size=9)
