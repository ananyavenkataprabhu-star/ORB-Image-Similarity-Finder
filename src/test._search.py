
import os
from search import search

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_dir = os.path.join(BASE_DIR, "dataset")

query_path = os.path.join(
    dataset_dir, "Buildings", "build_07.jpg"
)

results = search(query_path, dataset_dir)

print("\nSEARCH RESULTS")
print("----------------------------")

if not results:
    print("No similar images found.")
else:
    for i, (path, inliers) in enumerate(results, start=1):
        print(f"{i}. {os.path.basename(path)} | Inliers: {inliers}")
