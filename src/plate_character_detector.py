import cv2


class PlateCharacterDetector:

  def __init__(self):
    self.image_bgr = None
    self.image_gray = None
  
  def load_image(self, file_path):
    self.image_bgr = cv2.resize(cv2.imread(file_path), (240, 80))
    self.image_gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
  
  def create_kernel(self, size):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

  def apply_gaussian_blur(self, image_gray, kernel_size):
    return cv2.GaussianBlur(image_gray, kernel_size, 0)

  def apply_thresholding(self, image_gray):
    return cv2.threshold(image_gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

  def apply_erosion(self, image_bw, kernel):
    return cv2.morphologyEx(image_bw, cv2.MORPH_ERODE, kernel)

  