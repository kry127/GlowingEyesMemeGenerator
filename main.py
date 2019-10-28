import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from utility import EyeglowMemeConverter, pil_to_opencv

options = {
    "random_hue": False,
    "eyeglow_path": "eye_1.png",
    "faceglow_path": "face_1.png",
}

def main():

    cap = cv2.VideoCapture(0)
    eyeglow_converter = EyeglowMemeConverter(parameters=options)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_pil = eyeglow_converter.add_glow_iopencv(img)
        img = pil_to_opencv(im_pil)



        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()