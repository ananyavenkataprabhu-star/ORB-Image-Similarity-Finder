
import cv2

# Load an image
image = cv2.imread("dataset/test.jpg")

if image is None:
    print("Could not load image. Check the image path.")
else:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=1500)

    # Detect keypoints and calculate descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw detected keypoints
    output = cv2.drawKeypoints(
        image, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    print("Number of keypoints:", len(keypoints))

    # Display the result
    cv2.imshow("ORB Keypoints", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
