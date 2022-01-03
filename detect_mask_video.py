# Gerekenli modüllerin dahil edilmesi
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# maske tahmin fonksiyonu
def detect_and_predict_mask(frame, faceNet, maskNet):
	# kameradan okunan görüntüyü bir blob resim haline getiriyoruz
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# oluşturulan resmi faceNet ağına göndeririyoruz
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# resimde tespit edilen yüzlerin koordinatları alınıyor daha sonra bu tespitler üzerinden maske tahmini yapılacak
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		# yapılan tespitin olaslığının alınması
		confidence = detections[0, 0, i, 2]

		# olasılık yüzde elliden büyükse hesaplama başlar
		if confidence > 0.5:
			# yüzün etrafında bir kutu çizmek için koordinatların alınması
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# kutu sınırları frame sınırlarını aşmamalı
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# çizilen kutudan yüz çıkartılır ve BGR2RGB dönüşümü yapılır
			# 224x224 olacak şekilde resize edilir çünkü model bu biçimde bir input ile çalışıyor
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# yüz ve koordinatları döngünün üzerinde oluşturulan dizilere eklenir
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# eğer bir yüz tespit edilmişse maske tespitine başlanır
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# yüzün koordinatları ve maske olup olmadığı değeri geri döndürülür
	return (locs, preds)

# yüz tespiti için hazır modelin yüklenmesi
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# maske tespit modelinin yüklenmesi
maskNet = load_model("mask_detector.model")

# video başlatılıyor
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# video stream'i üzerindeki tüm frame'ler üzerinde döngü çalışır
while True:
	#frame okumak
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# frame'deki yüzleri tespit et ve maske takıp takmadıklarına karar ver
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# tespit edilen yüzler üzerinde döngü
	for (box, pred) in zip(locs, preds):

		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# maske değerlerine göre çizilecek çerçeve rengi ve etiketin belirlenmesi
		if mask > withoutMask and abs(mask-withoutMask) > 0.99:
			print(abs(mask-withoutMask))
			label = "Maske Takili"
			color = (0, 255, 0)
		else:
			label = "Maske Takili Degil"			
			color = (0, 0, 255)
		
		# etikete maske oranının da eklenmesi
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# frame üzerine çerçeve ve etiketin eklenmesi işlemi
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# yeni çizilen frame'in gösterilmesi
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# çarpı işaretine basılınca pencerenin kapatılması
	if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) <1:
		break

cv2.destroyAllWindows()
vs.stop()