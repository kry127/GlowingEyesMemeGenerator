import cv2
import numpy as np
from PIL import Image

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

pil_eyeglow = Image.open("eye_1.png")
pil_faceglow = Image.open("face_1.png")

def opencv_to_pil(img):
    # move PIL -> OpenCV
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def pil_to_opencv(im_pil):
    return np.asarray(im_pil)


opencv_eyeglow = pil_to_opencv(pil_eyeglow)
opencv_faceglow = pil_to_opencv(pil_faceglow)


def randomize_hue_eyeglow():
    hue = np.random.randint(0, 360)/360.0
    new_img = Image.fromarray(shift_hue(opencv_eyeglow, hue), 'RGBA')
    return new_img

def add_glow(img_opencv, img_pil, parameters=None):
    gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3)

    for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + int(h/2), x:x + w]

        img_pil_eyeglow = pil_eyeglow
        if parameters.get("random_hue", False):
            img_pil_eyeglow = randomize_hue_eyeglow()
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # info about conversions:
            # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/48602446
            # paste red eyes
            eg_w, eg_h = img_pil_eyeglow.size
            dim = (x + ex + int((ew - eg_w) / 2), y + ey + int((eh - eg_h) / 2))
            img_pil.paste(img_pil_eyeglow, dim, img_pil_eyeglow)

    return img_pil

default_glowing_parameters = {
    "random_hue": True
}

def add_glow_iopencv(img_opencv, parameters=None):
    if parameters is None:
        parameters = default_glowing_parameters
    im_pil = opencv_to_pil(img_opencv)
    return add_glow(img_opencv, im_pil, parameters=parameters)


def add_glow_ipil(im_pil, parameters=None):
    if parameters is None:
        parameters = default_glowing_parameters
    img_opencv = pil_to_opencv(im_pil)
    return add_glow(img_opencv, im_pil, parameters=parameters)


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb

def hue_example():
    img = Image.open('tweeter.png').convert('RGBA')
    arr = np.array(img)

    green_hue = (180-78)/360.0
    red_hue = (180-180)/360.0

    new_img = Image.fromarray(shift_hue(arr,red_hue), 'RGBA')
    new_img.save('tweeter_red.png')

    new_img = Image.fromarray(shift_hue(arr,green_hue), 'RGBA')
    new_img.save('tweeter_green.png')