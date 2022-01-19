import cv2
import matplotlib.pyplot as plt
import math
import pickle
from skimage import data, color
from sklearn.datasets import fetch_lfw_people
import numpy as np

imgs_to_use = ['camera', 'text', 'coins', 'moon',
                   'page', 'clock', 'immunohistochemistry',
                   'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)())
          for name in imgs_to_use]

faces = fetch_lfw_people()
positive_patches = faces.images

# POMOCNE FUNKCIJE

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')
    plt.show()

def scale(size):
    first = size[0]
    addedTen = first + 10
    x = math.floor(size[1] * addedTen / first)
    return (addedTen, x)

def scale_to_normal_size(image):
    if image.shape[0] < image.shape[1]:
      h = math.floor(300 * image.shape[0] / image.shape[1])
      return cv2.resize(image, (300, h))
    else:
      h = math.floor(300 * image.shape[1] / image.shape[0])
      return cv2.resize(image, (h, 300))

def get_hog(size):

    img = positive_patches[0]
    img = cv2.resize(img, size)
    img = img.astype(np.uint8)

    nbins = 9 # broj binova
    cell_size = (8, 8) # broj piksela po celiji
    block_size = (3, 3) # broj celija po bloku

    return cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                        img.shape[0] // cell_size[0] * cell_size[0]),
                              _blockSize=(block_size[1] * cell_size[1],
                                          block_size[0] * cell_size[0]),
                              _blockStride=(cell_size[1], cell_size[0]),
                              _cellSize=(cell_size[1], cell_size[0]),
                              _nbins=nbins)


def classify_window(window, clf, hog):
    features = hog.compute(window).reshape(1, -1)
    return clf.predict_proba(features)[0][1]

def process_image(image, clf, hog, step_size, window_size=(85, 105)):
    best_score = 0
    best_window = None
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
          this_window = (y, x)  # zbog formata rezultata
          window = image[y:y + window_size[1], x:x + window_size[0]]
          if window.shape == (window_size[1], window_size[0]):
              score = classify_window(window, clf, hog)
              if score > best_score:
                  best_score = score
                  best_window = this_window
                  w = window
    return best_score, best_window, w

def get_best_image(scores):
    print('u get best image sam')
    max_score = 0
    max_score_tupple = None
    for score_tupple in scores:
        if score_tupple[0] > max_score:
           max_score = score_tupple[0]
           max_score_tupple = score_tupple
    return max_score_tupple

# FORMIRANJE MODELA

clf_hog = []

def load_files():
    for index in range(10):
        f1=open("weights/svcs/svc" + str(index) + ".txt",'r+b')
        clf = pickle.load(f1)
        f1.close()
        f2=open("weights/sizes/size"+ str(index) + ".txt",'r+b')
        size = pickle.load(f2)
        f2.close()
        hog = get_hog(size)
        clf_hog.append((clf, hog, size))

def get_scores(itest):
    scores = []
    load_files()
    for x in clf_hog:
        score, score_window, w = process_image(itest, x[0], x[1], step_size=10, window_size=x[2])
        scores.append((score, score_window, w, x[2]))
    return scores