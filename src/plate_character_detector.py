import cv2


class PlateCharacterDetector:

    def __init__(self):
        self.image_bgr = None
        self.image_gray = None
        self.digit_width = 32
        self.digit_height = 64

    def load_image(self, file_path=None, image=None):
        if file_path is None and image is None:
            raise Exception("No image provided")

        if file_path is not None:
            self.image_bgr = cv2.resize(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB), (240, 80))
        else:
            self.image_bgr = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (240, 80))

        self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

    def detect_characters(self):
        character_rois = self.image_bgr.copy()
        crop_characters = []
        image_bw, image_erode = self._preprocessing()
        contours, _ = cv2.findContours(
            image_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in self.sort_contours(contours=contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            char_size_ratio = h / w
            if 1.1 <= char_size_ratio <= 6:
                char_height_ratio = h / self.image_bgr.shape[0]
                if 0.35 <= char_height_ratio <= 0.8:
                    cv2.rectangle(character_rois, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)

                    char = image_bw[y:(y + h), x:(x + w)]
                    width_padding = int(char_size_ratio * w * 0.1)
                    char = cv2.copyMakeBorder(
                        char, 2, 2, width_padding, width_padding, cv2.BORDER_CONSTANT, (0, 0, 0))
                    char = cv2.resize(char, dsize=(
                        self.digit_width, self.digit_height))
                    _, char = cv2.threshold(
                        char, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    crop_characters.append(char)

        return character_rois, crop_characters

    def sort_contours(self, contours, reverse=False):
        i = 0
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                                 key=lambda b: b[1][i], reverse=reverse))
        return contours

    def _preprocessing(self):
        image_blur = self.apply_gaussian_blur(
            image_gray=self.image_gray, kernel_size=(5, 5))
        image_bw = self.apply_thresholding(image_gray=image_blur)
        image_erode = self.apply_erosion(
            image_bw=image_bw, kernel=self.create_kernel(3))

        return image_bw, image_erode

    def create_kernel(self, size):
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    def apply_gaussian_blur(self, image_gray, kernel_size):
        return cv2.GaussianBlur(image_gray, kernel_size, 0)

    def apply_thresholding(self, image_gray):
        return cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def apply_erosion(self, image_bw, kernel):
        return cv2.morphologyEx(image_bw, cv2.MORPH_ERODE, kernel)
