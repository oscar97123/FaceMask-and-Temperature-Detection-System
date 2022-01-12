# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import serial
import time 
from linenotify import linenotify
import subprocess
#Setting up your arduino
arduino = serial.Serial('/dev/cu.usbmodem14501',9600)

playerProcess= None
lowConfidence = 0.75

isEND = False

def cal_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hrs = int(elapsed_time / 60 / 60)
    elapsed_mins = int((elapsed_time - (elapsed_hrs * 60 * 60)) / 60)
    elapsed_secs = float(elapsed_time - (elapsed_hrs * 60 * 60) - (elapsed_mins * 60))
    return elapsed_hrs, elapsed_mins, elapsed_secs

#face detectinon function
def detectAndPredictMask(frame, faceNet, maskNet):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > lowConfidence:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
vs = VideoStream(src=0).start()
lastime = time.time()
# loop over the frames from the video stream
data = False
tempLimit = 37.5

while True:
	_, _, elapsed_secs = cal_time(lastime, time.time())
	if not isEND and elapsed_secs > 1.0:
		isEND = False

	# grab the frame from the threaded video stream and resize it to have a maximum width of 900 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=900)

	# detect faces in the frame and determine if they are wearing a face mask or not
	(locs, preds) = detectAndPredictMask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask"  else (0, 0, 255) 
	
		while arduino.in_waiting:          # 若收到序列資料…
			data_raw = arduino.readline()  # 讀取一行
			data = data_raw.decode()   # 用預設的UTF-8解碼
			print('接收到的資料：', data)

		text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		if data :
			color = (0, 0, 255)  if float(data) > tempLimit else color
			text = "{}: {:.2f}%, Temp: {}*C".format(label, max(mask, withoutMask) * 100, data)
	
		# display the label and bounding box rectangle on the output frame
		cv2.putText(frame, text, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


		if not data:
			continue

		if label =="Mask" and float(data) <= tempLimit: # normal
		# if label =="Mask": # normal
			print("ACCESS GRANTED")

		else:
			print("ACCESS DENIED")
			msg = ""
			# abnormal	
			

			can_play_music = True if(playerProcess == None or playerProcess.poll() == 0) else False
			if label =="Mask" and float(data) > tempLimit:
				msg = f"溫度過高 Temparature: {data}"
				if can_play_music:
					playerProcess = subprocess.Popen(args="afplay ./temputure_too_high.mp3",shell=True)

			elif label !="Mask" and float(data) <= tempLimit:
				msg = f"沒有正確配戴口罩"
				if can_play_music:
					playerProcess =subprocess.Popen(args="afplay ./please_wear_mask.mp3",shell=True)
			elif label !="Mask" and float(data) > tempLimit:
				msg = f"沒有正確配戴口罩 溫度過高 Temparature: {data}"
				if can_play_music:
					playerProcess = subprocess.Popen(args="afplay ./please_wear_mask__temputure_too_high.mp3",shell=True)
			cv2.imwrite('frame.jpg',frame)
			if not isEND:
				linenotify(msg=msg, file=open('frame.jpg','rb'))
				isEND = True
				lastime = time.time()

	# show the output frame
	cv2.imshow("FaceMask Detection -- q to quit", frame)

	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
# vs.stop()
vs.stream.release()  # Stop recording