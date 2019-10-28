import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


from utility import add_glow_ipil

input_path = "face2.jpg"
output_path = "khalanskiy.png"
options = {
    "random_hue": True
}

def main():
    original = Image.open(input_path)
    #find and substitude all eyes
    add_glow_ipil(original, parameters=options)
    original.save(output_path)


if __name__ == "__main__":
    main()