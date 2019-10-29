import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


from opencv.utility import EyeglowMemeConverter, pil_to_opencv, shift_hue


def convert_image(input_file, output_file, options):
    eyeglow_converter = EyeglowMemeConverter(parameters=options)

    original = Image.open(input_file)
    #find and substitude all eyes
    eyeglow_converter.add_glow_ipil(original)
    original.save(output_file)

def make_collage(input_file, output_file, options):
    eyeglow_converter = EyeglowMemeConverter(parameters=options)

    original = Image.open(input_file)
    #find and substitude all eyes
    eyeglow_converter.add_glow_ipil(original)
    opencv_original = pil_to_opencv(original)

    hue_base = np.random.randint(0, 60)
    if not options.get("random_hue", False):
        hue_base = options.get("sparkle_hue", hue_base)

    orig_w, orig_h = original.size
    out_image = Image.new('RGB', (orig_w*3, orig_h*2))
    for k in range(3):
        converted = Image.fromarray(shift_hue(opencv_original, (hue_base + k*60)/360.0), 'RGB')
        out_image.paste(converted, (k*orig_w, 0))
    for k in range(3):
        converted = Image.fromarray(shift_hue(opencv_original, (hue_base + (k+3)*60)/360.0), 'RGB')
        out_image.paste(converted, (k*orig_w, orig_h))

    out_image.save(output_file)