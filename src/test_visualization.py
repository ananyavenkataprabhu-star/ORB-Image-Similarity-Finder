
import cv2
from matcher import visualize

image1 = r"dataset/Buildings/build_07.jpg"
image2 = r"dataset\Modified_Images\build_01_rotated.jpg"

vis, inliers, good = visualize(
    image1, image2, draw_box=False
)

print("Good matches:", good)
print("RANSAC inliers:", inliers)

cv2.imshow("ORB Feature Matching", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
