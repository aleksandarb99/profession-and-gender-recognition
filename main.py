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

    itest = load_image('test-images/4.jpg')
    itest = scale_to_normal_size(itest)
    scores = get_scores(itest)
    image = get_best_image(scores)
    display_image(image[2])

    # KLASIFIKACIJA POLA

    face = image[2]
    model = load_model("gender_model/model")
    face_crop = cv2.resize(face, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)
    conf = model.predict(face_crop)[0]
    print("Man: " + str(conf[0]))
    print("Woman: " + str(conf[1]))

