import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load the image in grayscale
image_path = 's1.png'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply thresholding
thresh_value = 128
max_value = 255
thresh_type = cv2.THRESH_BINARY

retval, binary_image = cv2.threshold(gray_image, thresh_value, max_value, thresh_type)

# Display the results
print(f"Threshold value used: {retval}")

plt.imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))
plt.title("Image")
plt.axis('off')
plt.show()