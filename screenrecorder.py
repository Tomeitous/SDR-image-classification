import cv2
import numpy as np
from PIL import Image,ImageGrab

from keras import models


#loading model
model = models.load_model('imageclassifier.h5')


while True:
        #screenrecord
        video = ImageGrab.grab(bbox=(0, 0, 800, 800))
        img_np = np.array(video)
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(frame, 'RGB')

        im = im.resize((256,256))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = int(model.predict(img_array)[0][0])

        #cv2.rectangle(frame, (0, 0), (100, 100), (255, 255, 0), 2)
        if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()