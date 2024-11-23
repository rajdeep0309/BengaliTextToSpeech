import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr
from gtts import gTTS
import os

# Load the image
image = cv2.imread('s1.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to smooth the image (optional)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Initialize EasyOCR reader
reader = easyocr.Reader(['bn'])

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store contours of thick text
thick_text_contours = []

# Set a minimum area threshold for what is considered "thick" text
min_area = 500  # This value may need adjustment based on the image resolution
min_aspect_ratio = 0.5  # Minimum width-to-height ratio for thick text

# Filter and store contours based on area and aspect ratio
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect_ratio = float(w) / h

    if area > min_area and aspect_ratio > min_aspect_ratio:
        thick_text_contours.append((x, y, w, h, contour))

print(thick_text_contours)
# Group contours by rows
rows = []
current_row = []
previous_y = -1

for (x, y, w, h, contour) in sorted(thick_text_contours, key=lambda c: c[1]):
    if previous_y == -1 or abs(y - previous_y) < 21:  # Adjust threshold as needed
        current_row.append((x, y, w, h, contour))
    else:
        rows.append(current_row)
        current_row = [(x, y, w, h, contour)]
    previous_y = y

if current_row:
    rows.append(current_row)

sentence = ""

# Process each row
for row in rows:
    row = sorted(row, key=lambda c: c[0])  # Sort by x-coordinate within the row
    for (x, y, w, h, contour) in row:
        # Draw the bounding box for the thick text region
        cv2.rectangle(image, (x - 10, y - 10), (x + w + 5, y + h + 5), (255, 0, 0), 2)
        cv2.putText(image, f'Thick Text', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Extract the region of interest (ROI) from the thresholded image
        headline_region_thresh = thresh[y - 15:(y - 10) + (h + 15), (x - 15):(x - 10) + w + 15]

        # Convert the thresholded region back to an RGB image
        headline_region_rgb = cv2.cvtColor(headline_region_thresh, cv2.COLOR_GRAY2RGB)

        # Use EasyOCR to read text from the thresholded region
        results = reader.readtext(headline_region_rgb,
                                  detail=0,                # Return only the text
                                  decoder='wordbeamsearch', # Use the wordbeamsearch decoder
                                  contrast_ths=0.7,         # Adjust contrast threshold
                                  adjust_contrast=0.5,      # Adjust contrast
                                  text_threshold=0.4,       # Text detection threshold
                                  low_text=0.3)             # Lower threshold for faint text

        # Join the results into a single string
        s = " ".join(results)
        sentence += " " + s
        print(s)

# Display the image with contours
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Thick Text Contours Detected')
plt.axis('off')
plt.show()

# Print the final sentence
print(sentence)

# Convert the final sentence to speech
language = 'bn'
myVoice = gTTS(text=sentence, lang=language)
myVoice.save("speech.mp3")
os.system("speech.mp3")
