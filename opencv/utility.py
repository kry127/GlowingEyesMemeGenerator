import cv2
import numpy as np
from PIL import Image, ImageDraw

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def opencv_to_pil(img):
    # move PIL -> OpenCV
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil

def pil_to_opencv(im_pil):
    return np.asarray(im_pil)


# https://stackoverflow.com/questions/7274221/changing-image-hue-with-python-pil
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


def amplify_saturation(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb



default_glowing_parameters = {
    "random_hue": True,
    "eyeglow_path": "eye_1.png",
    "faceglow_path": "face_1.png",
    "haar_scale_parameter": 1.1,
}

class EyeglowMemeConverter:
    def __init__(self, parameters=None):
        if parameters is None:
            self.parameters = default_glowing_parameters
        else:
            self.parameters=parameters
        self.pil_eyeglow = Image.open(self.parameters["eyeglow_path"])
        self.pil_faceglow = Image.open(self.parameters["faceglow_path"])
        self.opencv_eyeglow = pil_to_opencv(self.pil_eyeglow)
        self.opencv_faceglow = pil_to_opencv(self.pil_faceglow)
        self.haar_scale_parameter = self.parameters.get("haar_scale_parameter", 1.1)

    def randomize_hue_eyeglow(self):
        hue = np.random.randint(0, 360)/360.0
        new_img = Image.fromarray(shift_hue(self.opencv_eyeglow, hue), 'RGBA')
        return new_img

    def add_glow(self, img_opencv, img_pil):
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, self.haar_scale_parameter, 3)

        iDraw = ImageDraw.Draw(img_pil)

        for (x, y, w, h) in faces:
            #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            iDraw.rectangle([(x, y), (x + w, y + h)], None, outline=(0, 64, 255))

            roi_gray = gray[y:y + int(h/2), x:x + w]

            if self.parameters.get("show_face", False):
                # https://www.geeksforgeeks.org/python-pil-image-resize-method/
                face_resized = self.pil_faceglow.resize((w, h))
                fg_w, fg_h = face_resized.size
                dim = (x + int((w - fg_w) / 2), y + int((h - fg_h) / 2))
                img_pil.paste(face_resized, dim, face_resized)

            if not self.parameters.get("show_eyes", False):
                continue # skip eye detection

            img_pil_eyeglow = self.pil_eyeglow
            if self.parameters.get("random_hue", False):
                img_pil_eyeglow = self.randomize_hue_eyeglow()
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # info about conversions:
                # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/48602446
                # paste red eyes
                eg_w, eg_h = img_pil_eyeglow.size
                dim = (x + ex + int((ew - eg_w) / 2), y + ey + int((eh - eg_h) / 2))
                img_pil.paste(img_pil_eyeglow, dim, img_pil_eyeglow)

        return img_pil

    def add_glow_iopencv(self, img_opencv):
        im_pil = opencv_to_pil(img_opencv)
        return self.add_glow(img_opencv, im_pil)


    def add_glow_ipil(self, im_pil):
        img_opencv = pil_to_opencv(im_pil)
        return self.add_glow(img_opencv, im_pil)
