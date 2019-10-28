import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


from utility import EyeglowMemeConverter

input_path = "face.jpg"
output_path = "khalanskiy.png"


options = {
    "random_hue": False,
    "show_eyes": True,
    "show_face": True,
    "eyeglow_path": "eye_1.png",
    "faceglow_path": "face_1.png",
    "haar_scale_parameter": 1.4,
}

def main():
    eyeglow_converter = EyeglowMemeConverter(parameters=options)

    original = Image.open(input_path)
    #find and substitude all eyes
    eyeglow_converter.add_glow_ipil(original)
    original.save(output_path)


if __name__ == "__main__":
    main()