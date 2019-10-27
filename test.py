import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

lf = cv2.imread("eye_1.png")
user = cv2.imread("face.jpg")
gray_user = cv2.cvtColor(user, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_user, 1.3, 5)
for (x, y, w, h) in faces:
    roi_gray = gray_user[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        ew = ew + 400
        eh = eh + 400
        ey = ey - 202
        ex = ex - 206
        # cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)

        # resizing & paste the lf image on user
        roi_eye = user[y + ey:y + ey + eh, x + ex:x + ex + ew]
        resized_lensflare = cv2.resize(lf, (eh, ew))

        # making fg,bg and alpha
        fg = resized_lensflare.copy()
        alpha = fg.copy()
        bg = roi_eye.copy()

        # converting uint8 to float
        fg = fg.astype(float)
        bg = bg.astype(float)

        # Normalizing the alpha mask to keep intensity between 0 and 1
        alpha = alpha.astype(float) / 255

        fg = cv2.multiply(alpha, fg)
        bg = cv2.multiply(1.0 - alpha, bg)
        final = cv2.add(fg, bg)

        user[y + ey:y + ey + eh, x + ex:x + ex + ew] = final

cv2.imshow("image", user)

cv2.waitKey(0)
cv2.destroyAllWindows()