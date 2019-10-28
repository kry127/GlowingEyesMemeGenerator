import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from utility import pil_to_opencv, add_glow_iopencv

options = {
    "random_hue": False
}

def main():

    cap = cv2.VideoCapture(0)

    #return

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im_pil = add_glow_iopencv(img, parameters=options)
        img = pil_to_opencv(im_pil)



        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()