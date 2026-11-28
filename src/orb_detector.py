
import os
import cv2

# Load an image
image_path = r"C:\Users\venka\OneDrive\Documents\ORB-Image-Similarity-Finder\dataset\test2.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Could not load image. Check the image path.")
else:
    print("Full path:", os.path.abspath(image_path))

    # Resize image while preserving proportions
    display = cv2.resize(image, (960, 540))

    # Convert resized image to grayscale
    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect keypoints and calculate descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw keypoints on the SAME resized image
    output = cv2.drawKeypoints(
    display, keypoints, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
)

    print("Number of keypoints:", len(keypoints))

    # Display result
    cv2.imshow("Original Image", display)
    cv2.imshow("ORB Keypoints", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

