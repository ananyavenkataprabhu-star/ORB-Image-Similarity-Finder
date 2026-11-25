
import cv2
import os

# Load original image
image = cv2.imread("dataset/Buildings/build_07.jpg")

if image is None:
    print("Image not found! Check the image path.")
    exit()

# Create output folder
output_folder = "dataset/Modified_Images"
os.makedirs(output_folder, exist_ok=True)

# 1. Rotate
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite(output_folder + "/build_01_rotated.jpg", rotated)

# 2. Crop
h, w = image.shape[:2]
cropped = image[h//4:3*h//4, w//4:3*w//4]
cv2.imwrite(output_folder + "/build_01_cropped.jpg", cropped)

# 3. Resize
resized = cv2.resize(image, None, fx=0.9, fy=0.9)
cv2.imwrite(output_folder + "/build_01_resized.jpg", resized)

# 4. Darken
darkened = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
cv2.imwrite(output_folder + "/build_01_darkened.jpg", darkened)

print("4 modified images created!")
