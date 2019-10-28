import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


from opencv.utility import EyeglowMemeConverter


def convert_image(input_file, output_file, options):
    eyeglow_converter = EyeglowMemeConverter(parameters=options)

    original = Image.open(input_file)
    #find and substitude all eyes
    eyeglow_converter.add_glow_ipil(original)
    original.save(output_file)