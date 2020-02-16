import os
import re
import cv2
import imutils
import argparse
import pytesseract
import numpy as np
from PIL import Image
from nltk import edit_distance
from skimage.filters import threshold_local

from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files
        print(f)
	return 'got it'


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped


def binarize_image(image):
	image = cv2.imread(image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, gray)
	return filename


def transform_image(image):
	image = cv2.imread(image)
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	screenCnt = None
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points, then we can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
	if screenCnt is None:  # was not able to detect 4 corners, quit trying transforms
		return None

	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255

	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, imutils.resize(warped, height = 650))
	return filename


def ocr_text(filename):
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	return text


def text_parser(text):
	item_to_price = {}

	price_pattern = r"\d*\.\d{2}"
	item_pattern = r"\S{0,4}[A-Za-z]{2,20}"
	for line in text.splitlines():
		words = line.split()
		item = ""
		price = None

		for word in words:
			word = word.replace(',', '.')
			if re.fullmatch(price_pattern, word):  # found price!
				price = word
			elif re.fullmatch(item_pattern, word):
				item += " " + word

		if price:
			if edit_distance(item.upper(), "TOTAL") <= 2:
				break

			if edit_distance(item.upper(), "SUBTOTAL") > 2:  # don't include
				item_to_price[item] = price
				print (item, "|", price)
			
	return item_to_price

if __name__ == "__main__":
    # # construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("--image", "-i", required=True, help="path to input image to be OCR'd")
	# args = vars(ap.parse_args())

	# filename = transform_image(args["image"])
	# if filename is None:
	# 	filename = binarize_image(args["image"])

	# text = ocr_text(filename)
	# item_to_price = text_parser(text)

	app.run(host='0.0.0.0', port=5000)

