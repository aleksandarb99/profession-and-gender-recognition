import numpy as np
import matplotlib.pyplot as plt
import cv2

def detect_person(img_path):

    image = plt.imread(img_path)

    # classes = None
    # with open('yolo-coco/coco.names', 'r') as f:
    #     classes = [line.strip() for line in f.readlines()]

    Width = image.shape[1]
    Height = image.shape[0]

    # read pre-trained model and config file
    net = cv2.dnn.readNet('yolo-coco/yolov3.weights', 'yolo-coco/yolov3.cfg')

    # create input blob
    # set input blob for the network
    net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # create bounding box
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

    cropped_images = []
    # check if is people detection
    for i in indices:
        box = boxes[i]
        if class_ids[i] == 0:
            h = round(box[3])
            w = round(box[2])
            x = round(box[0])
            y = round(box[1])
            x = round(x - ((h/2)-(w/2)))
            if x < 0:
                x = 0
            cropped_image = image[y:y+h, x:x + h]
            cropped_images.append(cropped_image)

    for i in cropped_images:
        plt.imshow(i)
        plt.show()

    return cropped_images