import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


from opencv.utility import EyeglowMemeConverter, shift_hue, pil_to_opencv

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
    opencv_original = pil_to_opencv(original)

    hue_base = np.random.randint(0, 60)

    orig_w, orig_h = original.size
    out_image = Image.new('RGB', (orig_w*3, orig_h*2))
    for k in range(3):
        converted = Image.fromarray(shift_hue(opencv_original, (hue_base + k*60)/360.0), 'RGB')
        out_image.paste(converted, (k*orig_w, 0))
    for k in range(3):
        converted = Image.fromarray(shift_hue(opencv_original, (hue_base + (k+3)*60)/360.0), 'RGB')
        out_image.paste(converted, (k*orig_w, orig_h))

    out_image.save(output_path)


if __name__ == "__main__":
    main()