from __future__ import print_function
from professionrecognition.functions import get_model, run_inference
from facedetection.functions import load_image, scale_to_normal_size, get_scores, get_best_image, display_image
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np

if __name__ == "__main__":
    # KLASIFIKACIJA PROFESIJA
    # picture = "test-images/2.jpg"
    #
    # model = get_model()
    #
    # run_inference(model,picture)

    # DETEKCIJA LICA

    itest = load_image('test-images/3.jpg')

    itest = scale_to_normal_size(itest)
    scores = get_scores(itest)
    image = get_best_image(scores)

    image_colorful = cv2.cvtColor(cv2.imread("test-images/3.jpg"), cv2.COLOR_BGR2RGB)
    image_colorful = scale_to_normal_size(image_colorful)

    img_crop = image_colorful[image[1][0]:image[1][0] + image[3][1], image[1][1]:image[1][1] + image[3][0]]  # prva koordinata je po visini (formalno red), druga po širini (formalo kolona)\n"
    display_image(img_crop)

    # KLASIFIKACIJA POLA

    model = load_model("gender_model/model")
    face_crop = cv2.resize(img_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)
    conf = model.predict(face_crop)[0]
    print("Man: " + str(conf[0]))
    print("Woman: " + str(conf[1]))

