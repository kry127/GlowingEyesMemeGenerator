import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

pil_eyeglow = Image.open("eye_1.png")

# PIL example:
#def pil_paste_transparent():
    #background = Image.open("face.jpg")
    #foreground = Image.open("eye_1.png")
    #background.paste(foreground, (0, 0), foreground)
    #background.save("khalanskiy.png")

def main():

    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    #return

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # info about conversions:
                # https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format/48602446
                # move PIL -> OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                # paste red eyes
                eg_w, eg_h = pil_eyeglow.size
                dim = (x+ex+int((ew - eg_w)/2), y+ey+int((eh-eg_h)/2))
                im_pil.paste(pil_eyeglow, dim, pil_eyeglow)
                # move PIL -> OpenCV
                img = np.asarray(im_pil)



        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()