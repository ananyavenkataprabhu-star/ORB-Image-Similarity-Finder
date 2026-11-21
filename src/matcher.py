"""Shared core: feature extraction (with pickle cache), matching, RANSAC, visualization.

Everything else (search, scanner, evaluate, app) imports from here.
Supports two detectors: "orb" (default) and "akaze" (for the comparison).
"""

import hashlib
import os
import pickle

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")    #image formats recognised
CACHE_DIR = "cache"     #where previously extracted features are stored
DUP_THR, SIM_THR = 40, 15   #thresholds for duplicate and similar images


def list_images(folder):
    return sorted(
        os.path.join(folder, f)     #combines the folder path with the file name 
        for f in os.listdir(folder)     #lists all files in the folder
        if f.lower().endswith(IMG_EXT) 
    )


#loading an image and converting it to grayscale, resizing if necessary
def load_gray(path, max_side=1000):
    """Read an image, shrink it if huge, return (color, gray)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]    #getting image dimensions
    scale = max_side / max(h, w)    #calculating resize factor
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def make_detector(method="orb"): #default method is ORB, but can also use AKAZE if specified
    if method == "akaze":
        if not hasattr(cv2, "AKAZE_create"):
            raise RuntimeError("AKAZE is not available in this OpenCV build "
                               "(use: pip install opencv-python==4.10.0.84)")
        return cv2.AKAZE_create()
    return cv2.ORB_create(nfeatures=1500)


def _pack(kp, des, shape):
    pts = np.float32([k.pt for k in kp]) if len(kp) else np.zeros((0, 2), np.float32)   #extracting coordinates of keypoints
    return {"pts": pts, "des": des, "shape": shape, "n": len(pts)}


def extract(path, method="orb", use_cache=True):
    """Keypoint coordinates + descriptors for one image, cached on disk."""
    key = hashlib.md5(
        f"{os.path.abspath(path)}|{os.path.getmtime(path)}|{method}".encode()
    ).hexdigest()
    cache_path = os.path.join(CACHE_DIR, key + ".pkl")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    _, gray = load_gray(path)
    kp, des = make_detector(method).detectAndCompute(gray, None)
    feat = _pack(kp, des, gray.shape)

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(feat, f)
    return feat


def match(f1, f2, ratio=0.75, min_good=10, ransac_thr=5.0):
    """Hamming matching + Lowe ratio test + RANSAC homography.

    Returns (inlier_count, good_count, H, inlier_matches).
    """
    if f1["des"] is None or f2["des"] is None or f1["n"] < 2 or f2["n"] < 2:
        return 0, 0, None, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(f1["des"], f2["des"], k=2)
    good = [p[0] for p in pairs
            if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_good:
        return 0, len(good), None, []

    src = np.float32([f1["pts"][m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([f2["pts"][m.trainIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr)
    if H is None or mask is None:
        return 0, len(good), None, []

    inliers = [m for m, ok in zip(good, mask.ravel()) if ok]
    return len(inliers), len(good), H, inliers


def classify(inliers, dup_thr=DUP_THR, sim_thr=SIM_THR):
    if inliers >= dup_thr:
        return "duplicate"
    if inliers >= sim_thr:
        return "similar"
    return "different"


def visualize(p1, p2, method="orb", ratio=0.75, draw_box=True):
    """Side-by-side image with match lines (+ bounding box). Returns (vis, inliers, good)."""
    img1, g1 = load_gray(p1)
    img2, g2 = load_gray(p2)
    det = make_detector(method)
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    n, good, H, inliers = match(_pack(k1, d1, g1.shape), _pack(k2, d2, g2.shape), ratio)

    right = img2.copy()
    if H is not None and draw_box:
        h, w = g1.shape
        box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(box, H)
        right = cv2.polylines(right, [np.int32(proj)], True, (0, 255, 0), 3)

    vis = cv2.drawMatches(
        img1, k1, right, k2, inliers, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis, n, good