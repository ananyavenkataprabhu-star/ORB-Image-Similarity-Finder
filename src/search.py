
import cv2
import numpy as np
import os

# -----------------------------------
# 1. PROJECT PATHS AND SETTINGS
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

QUERY_PATH = os.path.join(
    DATASET_DIR, "Buildings", "build_01.jpg"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ORB and matching settings
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

RATIO_THRESHOLD = 0.75
MIN_INLIERS = 7
RANSAC_THRESHOLD = 5.0

# -----------------------------------
# 2. LOAD QUERY IMAGE
# -----------------------------------

query = cv2.imread(QUERY_PATH)

if query is None:
    print("ERROR: Query image not found:", QUERY_PATH)
    raise SystemExit

query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

kp1, des1 = orb.detectAndCompute(query_gray, None)

if des1 is None or len(kp1) < 4:
    print("Not enough features in query image.")
    raise SystemExit

print("Query image loaded successfully.")
print("Query keypoints:", len(kp1))

# -----------------------------------
# 3. FIND DATASET IMAGES
# -----------------------------------

image_paths = []

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            path = os.path.join(root, file)

            if os.path.abspath(path) != os.path.abspath(QUERY_PATH):
                image_paths.append(path)

print("Dataset images found:", len(image_paths))

# -----------------------------------
# 4. MATCH AND VERIFY IMAGES
# -----------------------------------

results = []

for path in image_paths:

    image = cv2.imread(path)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(gray, None)

    if des2 is None or len(kp2) < 4:
        continue

    # Feature matching
    pairs = bf.knnMatch(des1, des2, k=2)

    good = [
        m for m, n in pairs
        if m.distance < RATIO_THRESHOLD * n.distance
    ]

    if len(good) < 4:
        continue

    # Convert matched keypoints to NumPy arrays
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    # RANSAC geometric verification
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        RANSAC_THRESHOLD
    )

    if H is None or mask is None:
        continue

    inliers = int(np.sum(mask))
    inlier_ratio = inliers / len(good)

    # Keep sufficiently strong verified matches
    if inliers >= MIN_INLIERS:
        results.append({
            "path": path,
            "image": image,
            "matches": len(good),
            "inliers": inliers,
            "ratio": inlier_ratio
        })

# -----------------------------------
# 5. RANK RESULTS
# -----------------------------------

results.sort(
    key=lambda x: (x["inliers"], x["ratio"]),
    reverse=True
)

print("\nTOP VERIFIED SIMILAR IMAGES")
print("--------------------------------")

for i, result in enumerate(results[:5], start=1):
    print(
        f"{i}. {os.path.relpath(result['path'], BASE_DIR)}"
        f" | Matches: {result['matches']}"
        f" | Inliers: {result['inliers']}"
        f" | Ratio: {result['ratio']:.2f}"
    )

# -----------------------------------
# 6. DISPLAY QUERY IMAGE
# -----------------------------------

cv2.imshow("Query Image", query)

# -----------------------------------
# 7. DISPLAY TOP 5 RESULTS
# -----------------------------------

if not results:
    print("\nNo sufficiently strong matches found.")
else:
    for i, result in enumerate(results[:5], start=1):

        display = result["image"].copy()

        text = (
            f"Rank {i} | Inliers: {result['inliers']}"
        )

        cv2.putText(
            display,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        display = cv2.resize(display, (450, 300))

        cv2.imshow(f"Rank {i}", display)

    # Save the best matching image
    best = results[0]

    output_path = os.path.join(
        OUTPUT_DIR, "best_match.jpg"
    )

    cv2.imwrite(output_path, best["image"])

    print("\nBest match saved at:", output_path)

# -----------------------------------
# 8. CLOSE WINDOWS
# -----------------------------------

print("\nPress any key on an image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()

