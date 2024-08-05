import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
import easyocr

# Load the image
image = cv2.imread('s1.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
reader = easyocr.Reader(['bn'])

def filter_headline_contours(contours, image_height, min_contour_height_ratio=0.05):
    # Filter contours based on size and position
    headline_contours = []
    min_contour_height = min_contour_height_ratio * image_height

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > min_contour_height and y < image_height / 2:  # Large and near the top
            headline_contours.append((cnt, (x, y, w, h)))

    return headline_contours


#cv2.drawContours(image, contours, -1, (0,255,0), 3)
image_height = image.shape[0]
grouped_contours = filter_headline_contours(contours, image_height)
if grouped_contours:
    for idx, (_, (x, y, w, h)) in enumerate(grouped_contours):
        headline_region = image[y:y + h, x:x + w]
        results = reader.readtext(cv2.cvtColor(headline_region, cv2.COLOR_BGR2RGB))
        print(results)
        # Display the headline region
        plt.imshow(cv2.cvtColor(headline_region, cv2.COLOR_BGR2RGB))
        plt.title(f'Headline Region {idx}')
        plt.axis('off')
        plt.show()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Image")
plt.axis('off')
plt.show()
