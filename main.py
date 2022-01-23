from __future__ import print_function

import imutils as imutils

from professionrecognition.functions import get_model, run_inference
from persondetection.functions import detect_person
from facedetection.functions import load_image, scale_to_normal_size, get_scores, get_best_image, display_image
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

def get_faces(images):
    imgs_crop = []
    for img in images:
        image_normal_size = scale_to_normal_size(img)
        display_image(image_normal_size)
        image_normal_size_gray = cv2.cvtColor(image_normal_size, cv2.COLOR_BGR2GRAY)
        scores = get_scores(image_normal_size_gray)
        best_score = get_best_image(scores)

        img_crop = image_normal_size[best_score[1][0]:best_score[1][0] + best_score[2][1],
                   best_score[1][1]:best_score[1][1] + best_score[2][
                       0]]  # prva koordinata je po visini (formalno red), druga po širini (formalo kolona)\n"
        display_image(img_crop)
        imgs_crop.append(img_crop)
    return imgs_crop

if __name__ == "__main__":
    # DETEKCIJA OSOBE

    img_path = "test-images/23.jpg"
    images = detect_person(img_path)

    # KLASIFIKACIJA PROFESIJA

    model = get_model()
    for image in images:
        score = run_inference(model,image)
        print(score)

    # DETEKCIJA LICA

    imgs_crop = get_faces(images)

    # KLASIFIKACIJA POLA

    model = load_model("gender_model/model")

    for img_crop in imgs_crop:
        face_crop = cv2.resize(img_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = model.predict(face_crop)[0]
        print("Man: " + str(conf[0]))
        print("Woman: " + str(conf[1]))

