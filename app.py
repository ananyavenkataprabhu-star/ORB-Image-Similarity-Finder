
import os
import sys
import tempfile

import cv2
import streamlit as st

# Allow imports from the src folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.insert(0, SRC_DIR)

from search import search
from matcher import visualize, classify

# Page configuration
st.set_page_config(
    page_title="ORB Image Finder",
    page_icon="🔍",
    layout="wide"
)

st.title("Duplicate Image & Similarity Finder")
st.write("Find similar images using ORB feature detection and matching.")

# Dataset settings
dataset_dir = os.path.join(BASE_DIR, "dataset")

if not os.path.isdir(dataset_dir):
    st.error("Dataset folder not found.")
    st.stop()

# Image upload
st.subheader("Upload Query Image")

uploaded_file = st.file_uploader(
    "Choose an image to search for",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Query Image", width=300)

    if st.button("Search Images", type="primary"):

        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            query_path = temp_file.name

        with st.spinner("Searching your dataset..."):

            try:
                results = search(
    query_path,
    dataset_dir,
    top_n=6,
    query_name=uploaded_file.name
)

                st.session_state["results"] = results
                st.session_state["query_path"] = query_path

            except Exception as e:
                st.error(f"Search failed: {e}")

    # Display search results
    if "results" in st.session_state:

        results = st.session_state["results"]

        st.subheader("Search Results")

        if not results:
            st.warning("No sufficiently similar images found.")

        else:
            cols = st.columns(3)

            for i, (path, inliers) in enumerate(results):

                with cols[i % 3]:

                    st.image(
                        path,
                        caption=f"Rank {i + 1}: {os.path.basename(path)}",
                        use_container_width=True
                    )

                    st.write(f"Inliers: **{inliers}**")

                    st.write(
                        f"Classification: **{classify(inliers)}**"
                    )

            # Feature matching visualisation
            st.subheader("Feature Matching Visualisation")

            image_options = {
                os.path.basename(path): path
                for path, _ in results
            }

            selected_image = st.selectbox(
                "Choose an image to visualize matches",
                list(image_options.keys())
            )

            if st.button("Show Feature Matches"):

                selected_path = image_options[selected_image]

                vis, inliers, good = visualize(
                    st.session_state["query_path"],
                    selected_path
                )

                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

                st.image(
                    vis_rgb,
                    caption=f"{inliers} verified inliers out of {good} good matches",
                    use_container_width=True
                )
