

import cv2
import numpy as np
import os
import csv

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
MIN_INLIERS = 7
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
                "matches":len(good),
                "inliers": inliers,
                "ratio": inlier_ratio
            })

    # Rank results
    results.sort(
        key=lambda x: (x["inliers"], x["ratio"]),
        reverse=True
    )

    # Generate CSV report
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs"
    )
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(
        output_dir, "similarity_report.csv"
    )

    with open(report_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Rank",
            "Query Image",
            "Matched Image",
            "Good Matches",
            "Inliers",
            "Inlier Ratio"
        ])

        for rank, result in enumerate(results[:top_n], start=1):
            writer.writerow([
                rank,
                os.path.basename(query_path),
                os.path.relpath(result["path"], dataset_dir),
                result["matches"],
                result["inliers"],
                round(result["ratio"], 3)
            ])

    print("CSV report generated:", report_path)


    # Return paths and inlier counts for Streamlit
    return [
        (result["path"], result["inliers"])
        for result in results[:top_n]
    ]
