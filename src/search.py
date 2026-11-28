

import cv2
import numpy as np
import os

# ORB and matching settings
MIN_INLIERS = 15
RANSAC_THRESHOLD = 5.0


def search(query_path, dataset_dir, top_n=6,
           method="orb", ratio=0.75, exclude_self=True,
           query_name=None):

    if method.lower() != "orb":
        raise ValueError("This project uses ORB only.")

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Load query image
    query = cv2.imread(query_path)

    if query is None:
        raise ValueError(f"Could not load query image: {query_path}")

    query_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(query_gray, None)

    if des1 is None or len(kp1) < 4:
        return []

    # Find dataset images
    image_paths = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                path = os.path.join(root, file)

                if exclude_self and os.path.abspath(path) == os.path.abspath(query_path):
                    continue

                image_paths.append(path)

    # Match and verify images
    results = []

    for path in image_paths:

        image = cv2.imread(path)

        if image is None:
            continue

        # Skip image with the same uploaded filename
        if query_name and os.path.basename(path).lower() == query_name.lower():
            continue

        # Skip an exact pixel-for-pixel copy
        if image.shape == query.shape and np.array_equal(image, query):
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des2 is None or len(kp2) < 4:
            continue

        # Feature matching
        pairs = bf.knnMatch(des1, des2, k=2)

        good = [
            pair[0] for pair in pairs
            if len(pair) == 2
            and pair[0].distance < ratio * pair[1].distance
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

        print(
            os.path.basename(path),
            "| Good matches:", len(good),
            "| Inliers:", inliers,
            "| Inlier ratio:", round(inlier_ratio, 3)
        )

        if inliers >= MIN_INLIERS:
            results.append({
                "path": path,
                "inliers": inliers,
                "ratio": inlier_ratio
            })

    # Rank results
    results.sort(
        key=lambda x: (x["inliers"], x["ratio"]),
        reverse=True
    )

    # Return paths and inlier counts for Streamlit
    return [
        (result["path"], result["inliers"])
        for result in results[:top_n]
    ]
