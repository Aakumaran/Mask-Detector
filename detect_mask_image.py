# USAGE
# python detect_mask_image.py --image examples/example_01.png

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

image=cv2.imread('image.jpg')

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# initialize our list of faces, their corresponding locations,
# and the list of predictions from our face mask network
faces = []
locs = []
preds = []
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(30, 30))
for (x, y, w, h) in faces:
	face = image[y:y+h, x:x+w]
	face = cv2.resize(face, (224, 224))
	face = img_to_array(face)
	face = preprocess_input(face)
	face = np.expand_dims(face, axis=0)
	preds = maskNet.predict(face)
	label = "Mask" if preds[0][0] > preds[0][1] else "No Mask"
	color = (0, 255, 0) if label = "Mask" else (0, 0, 255)

	# include the probability in the label
	label = "{}: {:.2f}%".format(label, max(preds[0][0], preds[0][1]) * 100)

	# display the label and bounding box rectangle on the output
	# frame
	cv2.putText(image, label, (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
