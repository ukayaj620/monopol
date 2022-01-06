# Monopol

Indonesian Car License Plate (Indonesia Mobil Nomor Polisi) Character Recognition using Tensorflow, Keras and OpenCV.


## Background

This application is created to accomplish computer vision course final project.


## Methodology

The method that we use consist of three steps:
1. **Image preprocessing**, a process to seperate the characters of license plate from the plate background 
by using image processing technique starting from binarization, bluring, thresholding, and morphology.
2. **License Plate Character Detection**, a process to detect and segment the characters from the preprocessed 
plate image by finding it's contours. After the characters are segmented, then the characters will be cropped
an safe as an array of cropped character images.
3. **License Plate Character Recognition**, a process to recognize a character from cropped image processed before 
using the deep learning approach. The deep learning model that used in this project is Convolutional Neural Network 
and the architecture used is MobileNetV2 using the customize top layer (fully-connected layer). The output of the top
layer is the confidence rate of every single character classes (A-Z, 0-9).


## Run Locally

Step-by-step to run this program.

1. Clone or fork this repo.
2. Create a python virtual environment using virtualenv, conda, or miniconda. It's your choice.
3. Install all packages written inside ```requirements.txt``` by running ```pip install -r requirements.txt```.
4. Run ```flask run```
5. Open browser and access http://127.0.0.1:5000


## Contributor

1. Jayaku Briliantio
2. Ferdy Nicolas
3. Jason Alexander
4. Martien Junaedi
5. Kevin Hosea


## License

[MIT](./LICENSE)


## Paper References

| **Title** | **Link** |
| --------- | -------- |
| MobileNetV2: Inverted Residuals and Linear Bottlenecks | [arXiv](https://arxiv.org/abs/1801.04381) |
