import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature
from google.colab import files

# Upload the image
uploaded = files.upload()

# Assume only one image is uploaded
filename = list(uploaded.keys())[0]

# Read the image
I = io.imread(filename)

# Apply Gaussian filter
Iblur = filters.gaussian(I, sigma=2, multichannel=True)

# Convert to grayscale
gray = color.rgb2gray(Iblur)

# Smooth out the image to remove irregular patches
gray_smooth = filters.median(gray, selem=np.ones((10, 10)))

# Perform Canny edge detection
edges = feature.canny(gray_smooth, sigma=1)

# Create a composite image with original image overlaid on edge detection result
composite_image = np.zeros_like(I)
composite_image[:,:,0] = np.maximum(I[:,:,0], edges)
composite_image[:,:,1] = np.maximum(I[:,:,1], edges)
composite_image[:,:,2] = np.maximum(I[:,:,2], edges)

# Display the original, smoothed grayscale, edge detection, and composite images
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(I)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')
axes[0, 1].imshow(gray_smooth, cmap='gray')
axes[0, 1].set_title('Smoothed Grayscale Image')
axes[0, 1].axis('off')
axes[1, 0].imshow(edges, cmap='gray')
axes[1, 0].set_title('Canny Edge Detection')
axes[1, 0].axis('off')
axes[1, 1].imshow(composite_image)
axes[1, 1].set_title('Original Image + Edge Detection')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()