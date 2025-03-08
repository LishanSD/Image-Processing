import cv2
import numpy as np


# Function to load the image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from path: {image_path}")
    return image


# Function to convert the image to grayscale
def convert_to_grayscale(image):
    grayscale_image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return grayscale_image


# Function to enhance the contrast of the grayscale image
def enhance_contrast(gray):
  # Find the minimum and maximum pixel values
  min_val = gray.min()
  max_val = gray.max()

  enhanced_gray = ((gray - min_val) / (max_val - min_val)) * 255
  enhanced_gray = np.clip(enhanced_gray, 0, 255) # Clip the values to the range 0-255
  return enhanced_gray


# Function to apply the filter to the image
def apply_filter(image, filter_matrix):
    # Normalize the filter if necessary
    if np.sum(filter_matrix) != 0:
        filter_matrix = filter_matrix / np.sum(filter_matrix)
    
    filter_size = filter_matrix.shape[0] # Get the filter size

    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    padded_image = np.pad(image, ((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2)), mode='constant') # Create a padded image with zeros

    # Apply the filter using convolution
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.sum(padded_image[i:i+filter_size, j:j+filter_size] * filter_matrix)

    filtered_image = np.clip(filtered_image, 0, 255) # Clip the values to the range 0-255

    return filtered_image


# Function to compute the RMS difference between two images
def compute_rms_difference(image1, image2):
    difference = image1.astype(np.float32) - image2.astype(np.float32)
    rms = np.sqrt(np.mean(difference ** 2)) 
    return rms



# Main code

# 1. Load the image
image = load_image('road39.png') 

# 2. Convert the image to grayscale format with 8bpp
grayscale_img = convert_to_grayscale(image)

# 3. Enhance the image contrast linearly
enhanced_gray = enhance_contrast(grayscale_img)

# Save the enhanced grayscale image as 'original.jpg'
cv2.imwrite('original.jpg', enhanced_gray)


# 4. Apply the filters

original_image = cv2.imread('original.jpg')

# Define the filters
Filter_A = np.array([[0, -1, -1, -1, 0],
                     [-1, 2, 2, 2, -1],
                     [-1, 2, 8, 2, -1],
                     [-1, 2, 2, 2, -1],
                     [0, -1, -1, -1, 0]])

Filter_B = np.array([[1, 4, 6, 4, 1],
                     [4, 16, 24, 16, 4],
                     [6, 24, 36, 24, 6],
                     [4, 16, 24, 16, 4],
                     [1, 4, 6, 4, 1]])

Filter_C = np.array([[5, 5, 5, 5, 5],
                     [5, 5, 5, 5, 5],
                     [5, 5, 5, 5, 5],
                     [5, 5, 5, 5, 5],
                     [5, 5, 5, 5, 5]])

Filter_D = np.array([[0, -1, -1, -1, 0],
                     [-1, 2, 2, 2, -1],
                     [-1, 2, 16, 2, -1],
                     [-1, 2, 2, 2, -1],
                     [0, -1, -1, -1, 0]])

# Apply each filter and save the result
filters = {'Filter_A': Filter_A, 'Filter_B': Filter_B, 'Filter_C': Filter_C, 'Filter_D': Filter_D}

for filter_name, filter_matrix in filters.items():
    filtered_image = apply_filter(original_image, filter_matrix)
    cv2.imwrite(f'{filter_name}.jpg', filtered_image)


# 5. Compute the RMS difference

original_image = cv2.imread('original.jpg')

# Compute and print the RMS difference for each filtered image
for filter_name in ['Filter_A', 'Filter_B', 'Filter_C', 'Filter_D']:
    filtered_image = cv2.imread(f'{filter_name}.jpg')  # Read the filtered image
    rms_difference = compute_rms_difference(original_image, filtered_image) # Compute the RMS difference
    print(f'RMS difference between original and {filter_name}: {rms_difference:.4f}')