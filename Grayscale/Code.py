import cv2
import numpy as np
import matplotlib.pyplot as pyplot

def convert_to_grayscale(image):
    height, width, channels = image.shape

    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            red = image[i, j, 0]
            green = image[i, j, 1]
            blue = image[i, j, 2]
            grayscale_image[i, j] = int(0.299 * red + 0.587 * green + 0.114 * blue)

    return grayscale_image

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from path: {image_path}")
    return image

# Load the color image
color_image = load_image('SrcImage.jpg') 

# Convert the image to grayscale
grayscale_img = convert_to_grayscale(color_image)

# Create subplots
fig, axs = pyplot.subplots(2, 3, figsize=(12, 8))

# Grid (1,1): Unprocessed grayscale image
axs[0, 0].imshow(grayscale_img, cmap='gray')
axs[0, 0].set_title('Unprocessed Grayscale')
cv2.imwrite('OPImage_1,1.jpg', grayscale_img)

# Grid (1,2): Negative image
negative_img = 255 - grayscale_img
axs[0, 1].imshow(negative_img, cmap='gray')
axs[0, 1].set_title('Negative Image')
cv2.imwrite('OPImage_1,2.jpg', negative_img)

# Grid (1,3): Increased brightness by 20%
bright_img = grayscale_img + 0.2 * grayscale_img
bright_img = np.clip(bright_img, 0, 255).astype(np.uint8)
axs[0, 2].imshow(bright_img, cmap='gray')
axs[0, 2].set_title('Increased Brightness')
cv2.imwrite('OPImage_1,3.jpg', bright_img)

# Grid (2,1): Reduced contrast
contrast_img = np.clip(grayscale_img, 125, 175)
axs[1, 0].imshow(contrast_img, cmap='gray')
axs[1, 0].set_title('Reduced Contrast')
cv2.imwrite('OPImage_2,1.jpg', contrast_img)

# Grid (2,2): Reduced gray level depth to 4bpp
four_bpp_img = (grayscale_img // 64) * 64
axs[1, 1].imshow(four_bpp_img, cmap='gray')
axs[1, 1].set_title('4-bit Grayscale')
cv2.imwrite('OPImage_2,2.jpg', four_bpp_img)

# Grid (2,3): Vertical mirror image
mirror_img = grayscale_img[:, ::-1]  # Reverse the columns
axs[1, 2].imshow(mirror_img, cmap='gray')
axs[1, 2].set_title('Vertical Mirror')
cv2.imwrite('OPImage_2,3.jpg', mirror_img)

# Save the entire subplot
pyplot.savefig('SubPlot.jpg')