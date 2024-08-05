import cv2
import numpy as np
from matplotlib import pyplot as plt


def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return image, contours


def filter_and_group_contours(contours, image_height, min_contour_height_ratio=0.05, max_distance=20):
    # Filter contours based on size and position, then group them
    min_contour_height = min_contour_height_ratio * image_height
    grouped_contours = []

    # Sort contours by their vertical position
    contours_sorted = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[1])

    while contours_sorted:
        current_contour = contours_sorted.pop(0)
        x, y, w, h = cv2.boundingRect(current_contour)

        # Find contours close to the current one
        group = [(current_contour, (x, y, w, h))]
        group_y_end = y + h

        i = 0
        while i < len(contours_sorted):
            next_contour = contours_sorted[i]
            nx, ny, nw, nh = cv2.boundingRect(next_contour)

            if abs(ny - group_y_end) < max_distance and abs(nx - x) < 2 * max_distance:
                group.append((next_contour, (nx, ny, nw, nh)))
                group_y_end = max(group_y_end, ny + nh)
                contours_sorted.pop(i)
            else:
                i += 1

        grouped_contours.append(group)

    return grouped_contours


def extract_text_blocks(image, grouped_contours):
    text_blocks = []

    for group in grouped_contours:
        x_min = min([cv2.boundingRect(cnt)[0] for cnt, _ in group])
        y_min = min([cv2.boundingRect(cnt)[1] for cnt, _ in group])
        x_max = max([cv2.boundingRect(cnt)[0] + cv2.boundingRect(cnt)[2] for cnt, _ in group])
        y_max = max([cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] for cnt, _ in group])

        text_block = image[y_min:y_max, x_min:x_max]
        text_blocks.append(text_block)

    return text_blocks


def main(image_path):
    # Preprocess the image and extract contours
    image, contours = preprocess_image(image_path)
    image_height = image.shape[0]

    # Filter and group contours to find text blocks
    grouped_contours = filter_and_group_contours(contours, image_height)

    # Extract and display text blocks
    if grouped_contours:
        for idx, text_block in enumerate(extract_text_blocks(image, grouped_contours)):
            # Display the text block
            plt.imshow(cv2.cvtColor(text_block, cv2.COLOR_BGR2RGB))
            plt.title(f'Text Block {idx}')
            plt.axis('off')
            plt.show()

            # Save the text block image
            text_block_image_path = f'text_block_{idx}.png'
            cv2.imwrite(text_block_image_path, text_block)

            print(f"Text block {idx} saved as {text_block_image_path}.")
            print("Please manually describe the text content of the saved image.")
    else:
        print("No text blocks detected.")


# Path to the newspaper image
image_path = 's1.png'

# Run the main function
main(image_path)
