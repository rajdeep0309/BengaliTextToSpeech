import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('s4.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to smooth the image (optional)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
# Initialize a list to store contours of thick text
thick_text_contours = []
# Set a minimum area threshold for what is considered "thick" text
min_area = 90  # This value may need adjustment based on the image resolution
min_aspect_ratio = 0.9 # Minimum width-to-height ratio for thick text

# Iterate through each contour
for contour in contours:
    # Calculate the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the area of the contour
    area = cv2.contourArea(contour)

    # Calculate the aspect ratio (width/height) of the contour
    aspect_ratio = float(w) / h

    # Filter contours based on area and aspect ratio
    if area > min_area and aspect_ratio > min_aspect_ratio:
        thick_text_contours.append(contour)
        # Optionally, draw the bounding box for the thick text region
        cv2.rectangle(image, (x-10, y-10), (x + w+5, y + h+5), (255, 0, 0), 2)
        cv2.putText(image, f'Thick Text', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



# Display the image with contours
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Thick Text Contours Detected')
plt.axis('off')
plt.show()

# Optional: Save the result to a file
cv2.imwrite('thick_text_contours_detected.png', image)
