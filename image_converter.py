import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

input_path = "face.jpg"
output_path = "khalanskiy.png"
eyeglow_path = "eye_1.png"

pil_eyeglow = Image.open(eyeglow_path)

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def opencv_to_pil(img):
    # move PIL -> OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def pil_to_opencv(im_pil):
    return np.asarray(im_pil)

def add_glow(im_pil):
    img = pil_to_opencv(im_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # info about conversions:
            # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/48602446
            # paste red eyes
            eg_w, eg_h = pil_eyeglow.size
            dim = (x + ex + int((ew - eg_w) / 2), y + ey + int((eh - eg_h) / 2))
            im_pil.paste(pil_eyeglow, dim, pil_eyeglow)

def main():
    original = Image.open(input_path)
    #find and substitude all eyes
    add_glow(original)
    original.save(output_path)


if __name__ == "__main__":
    main()