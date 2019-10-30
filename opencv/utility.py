import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def opencv_to_pil(img):
    # move PIL -> OpenCV
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

# create outline text
# https://stackoverflow.com/questions/41556771/is-there-a-way-to-outline-text-with-a-dark-line-in-pil
def draw_shadowed_text(draw, xy, text, textColor, font, adj, shadowColor = (0, 0, 0)):
    outline_amount = 3
    x, y = xy
    for adj in range(outline_amount):
        #move right
        draw.text((x-adj, y), text, font=font, fill=shadowColor)
        #move left
        draw.text((x+adj, y), text, font=font, fill=shadowColor)
        #move up
        draw.text((x, y+adj), text, font=font, fill=shadowColor)
        #move down
        draw.text((x, y-adj), text, font=font, fill=shadowColor)
        #diagnal left up
        draw.text((x-adj, y+adj), text, font=font, fill=shadowColor)
        #diagnal right up
        draw.text((x+adj, y+adj), text, font=font, fill=shadowColor)
        #diagnal left down
        draw.text((x-adj, y-adj), text, font=font, fill=shadowColor)
        #diagnal right down
        draw.text((x+adj, y-adj), text, font=font, fill=shadowColor)

        #create normal text on image
        draw.text((x,y), text, font=font, fill=textColor)

class EyeglowMemeConverter:
    def __init__(self, parameters=None):
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        self.pil_eyeglow = Image.open(self.parameters["eyeglow_path"]).convert("RGBA")
        self.pil_faceglow = Image.open(self.parameters["faceglow_path"]).convert("RGBA")
        self.opencv_eyeglow = pil_to_opencv(self.pil_eyeglow)
        self.opencv_faceglow = pil_to_opencv(self.pil_faceglow)
        self.face_cascade = cv2.CascadeClassifier(self.parameters["face_cascade"])
        self.eye_cascade = cv2.CascadeClassifier(self.parameters["eye_cascade"])
        self.glasses_cascade_path = cv2.CascadeClassifier(self.parameters["glasses_cascade_path"])
        self.sparkle_hue = self.parameters.get("sparkle_hue", None)
        self.haar_scale_parameter = self.parameters.get("haar_scale_parameter", 1.5)
        self.resize_to_box = self.parameters.get("resize_to_box", False)
        self.substitude_eyes = self.parameters.get("substitude_eyes", True)
        self.substitude_face = self.parameters.get("substitude_face", False)
        self.randomize_color = self.parameters.get("randomize_color", False)
        self.resize_ratio = self.parameters.get("resize_ratio", 1.0)
        self.meme_text = self.parameters.get("meme_text", "")
        self.meme_font = self.parameters.get("meme_font", "")

        self.meme_text_height_offset = 40
        self.meme_text_shade_repeat = 4

    def randomize_hue_eyeglow(self):
        hue = np.random.randint(0, 360)/360.0
        new_img = Image.fromarray(shift_hue(self.opencv_eyeglow, hue), 'RGBA')
        return new_img

    def colorize_hue_eyeglow(self):
        if not self.sparkle_hue:
            return self.pil_eyeglow
        new_img = Image.fromarray(shift_hue(self.opencv_eyeglow, self.sparkle_hue), 'RGBA')
        return new_img

    def add_glow(self, img_opencv, img_pil):
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, self.haar_scale_parameter, 3)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + int(h/2), x:x + w]

            if self.substitude_face:
                # https://www.geeksforgeeks.org/python-pil-image-resize-method/
                if self.resize_to_box:
                    scale = self.resize_ratio
                    face_resized = self.pil_faceglow.resize((int(w*scale), int(h*scale)))
                else:
                    face_resized = self.pil_faceglow
                fg_w, fg_h = face_resized.size
                dim = (x + int((w - fg_w) / 2), y + int((h - fg_h) / 2))
                img_pil.paste(face_resized, dim, face_resized)

            if not self.substitude_eyes:
                continue # skip eye detection

            img_pil_eyeglow_colorized  = self.pil_eyeglow
            if not self.sparkle_hue:
                img_pil_eyeglow_colorized = self.colorize_hue_eyeglow()
            if self.randomize_color:
                img_pil_eyeglow_colorized = self.randomize_hue_eyeglow()


            # simple eye processing
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:# resize to box if needed
                if self.resize_to_box:
                    scale = self.resize_ratio
                    img_pil_eyeglow = img_pil_eyeglow_colorized.resize((int(ew*scale), int(eh*scale)))
                else:
                    img_pil_eyeglow = img_pil_eyeglow_colorized
                # paste red eyes
                eg_w, eg_h = img_pil_eyeglow.size
                dim = (x + ex + int((ew - eg_w) / 2), y + ey + int((eh - eg_h) / 2))
                img_pil.paste(img_pil_eyeglow, dim, img_pil_eyeglow)

        img_pil = self.add_meme_text(img_pil)
        return img_pil

    def add_glow_iopencv(self, img_opencv):
        im_pil = opencv_to_pil(img_opencv)
        return self.add_glow(img_opencv, im_pil)


    def add_glow_ipil(self, im_pil):
        img_opencv = pil_to_opencv(im_pil)
        return self.add_glow(img_opencv, im_pil)

    def add_meme_text(self, img_pil):
        text = self.meme_text
        img_w, img_h = img_pil.size
        effective_width = img_w*0.90
        font = ImageFont.truetype(self.meme_font, 24)
        draw = ImageDraw.Draw(img_pil)
        splitted_text = []
        accum = ""
        for word in text.split():
            tmp_txt = accum + " " + word
            tw, th = font.getsize(tmp_txt)
            if tw > effective_width:
                splitted_text.append(accum)
                accum = word
            else:
                accum = tmp_txt
        splitted_text.append(accum)

        #print line by line splitted text
        o_height = 0
        for u in reversed(range(len(splitted_text))):
            line = splitted_text[u]
            tw, th = font.getsize(line)
            padding = max(th // 5, 2)
            o_height += th + padding
            k = u + 1
            pos = ((img_w - tw) // 2, img_h - o_height + padding // 2 - self.meme_text_height_offset)
            draw_shadowed_text(draw, pos, line, (255, 255, 255), font, 2)
            #draw.text(((img_w - tw) // 2, img_h - o_height + padding // 2), line, (0, 0, 0), font=font)

        return img_pil
