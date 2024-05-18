# Step 1: Upload the Image
from google.colab import files
uploaded = files.upload()
image_filename = next(iter(uploaded))  # Extract the uploaded image filename

# Step 2: Install and Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 3: Load and Process the Image
# Load the image
image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

# Preprocess the image: Thresholding
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Noise reduction: Morphological operations
kernel = np.ones((3, 3), np.uint8)
binary_image = cv2.erode(binary_image, kernel, iterations=1)
binary_image = cv2.dilate(binary_image, kernel, iterations=2)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Print contour areas for inspection
print("Contour areas:")
for contour in contours:
    area = cv2.contourArea(contour)
    print(area)

# Filter contours
filtered_contours = []
min_area = 5    # Minimum area of the pores
max_area = 100  # Maximum area of the pores
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        filtered_contours.append(contour)

# Draw contours (optional)
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 1)

# Save the result
output_filename = 'detected_pores.png'
cv2.imwrite(output_filename, output_image)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.title('Detected Pores')
plt.axis('off')
plt.show()