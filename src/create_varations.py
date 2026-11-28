
import cv2
import os

# Load original Taj Mahal image
image = cv2.imread("dataset/Buildings/build_16.jpg")

if image is None:
    print("Image not found! Check the image path.")
    exit()

# Create output folder
output_folder = "dataset/Modified_Images"
os.makedirs(output_folder, exist_ok=True)

# 1. Rotate by 15 degrees
h, w = image.shape[:2]
center = (w // 2, h // 2)

matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, matrix, (w, h))

cv2.imwrite(output_folder + "/build_16_rotated.jpg", rotated)

# 2. Crop moderately
cropped = image[h//3:2*h//3, w//3:2*w//3]

cv2.imwrite(output_folder + "/build_16_cropped.jpg", cropped)

print("2 Taj Mahal variations created!")
