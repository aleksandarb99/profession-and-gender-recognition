from io import open
import numpy as np
import json
from tensorflow.python.keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import cv2
from keras.preprocessing import image

JSON_PATH = "assets/idenprof_model_class.json"
input_shape = (224, 224, 3)
MODEL_PATH = "assets/idenprof_VGG16_053-0.726.h5"
CLASS_INDEX = None

def run_inference(model,picture):
    image_to_predict = cv2.resize(picture, (224, 224))
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)

    image_to_predict = preprocess_input(image_to_predict)

    prediction = model.predict(x=image_to_predict, steps=1)

    predictiondata = decode_predictions(prediction, top=int(5), model_json=JSON_PATH)

    maximum = predictiondata[0]
    for result in predictiondata:
        score = result[1] * 100
        if score > maximum[1] * 100:
            maximum = result
        print(str(result[0]), " : ", str(result[1] * 100))
    return maximum

def preprocess_input(x):
    x *= (1. / 255)
    return x

def decode_predictions(preds, top=5, model_json=""):
    global CLASS_INDEX
    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)
    return results

def get_model():
    model = Sequential(
        [Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
         Conv2D(64, (3, 3), activation='relu', padding='same'),
         MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
         Conv2D(128, (3, 3), activation='relu', padding='same'),
         Conv2D(128, (3, 3), activation='relu', padding='same', ),
         MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
         Conv2D(256, (3, 3), activation='relu', padding='same', ),
         Conv2D(256, (3, 3), activation='relu', padding='same', ),
         Conv2D(256, (3, 3), activation='relu', padding='same', ),
         Conv2D(256, (3, 3), activation='relu', padding='same', ),
         MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         Conv2D(512, (3, 3), activation='relu', padding='same', ),
         MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
         Flatten(),
         Dense(10, activation="softmax")])
    model.load_weights(MODEL_PATH)
    return model