# ==================================================
# Maize Disease Detection - Streamlit App 
# ==================================================

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os, json, pandas as pd, altair as alt
import gdown
import tensorflow as tf

url = "https://drive.google.com/file/d/11k3YVLExqOQaEHT1NXFh4s_cRjGwsko4/view?usp=sharing"
output = "model.h5"
gdown.download(url, output, quiet=False)

model = tf.keras.models.load_model("model.h5")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Maize Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
    body {font-family: 'Segoe UI', sans-serif;}
    .main-title { font-size: 2.4em; font-weight: 600; color: #1B4332; margin-bottom: 0.4em; }
    .sub-text { color: #4B5563; font-size: 1.1em; margin-bottom: 1.2em; }
    .result-box { padding: 1.2em; border-radius: 10px; border: 1px solid #E5E7EB; background-color: #F9FAFB; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Paths & constants
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, "maize_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")
IMG_SIZE = (224, 224)

# -------------------------------
# Load Model and Class Mapping
# -------------------------------
# -------------------------------
# Load Model and Class Mapping
# -------------------------------
@st.cache_resource
def load_trained_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path)

@st.cache_data
def load_class_mapping(path):
    import json, os
    default_mapping = {
        "Blight": 0,
        "Common Rust": 1,
        "Gray_Leaf_Spot": 2,
        "Healthy": 3
    }

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                mapping = json.load(f)
            return [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
        except Exception:
            return [k for k, v in sorted(default_mapping.items(), key=lambda item: item[1])]
    else:
        return [k for k, v in sorted(default_mapping.items(), key=lambda item: item[1])]

# Load them
model = load_trained_model(MODEL_PATH)
CLASS_NAMES = load_class_mapping(CLASS_INDICES_PATH)


# -------------------------------
# Initialize session state for page
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# -------------------------------
# Navigation Callback Functions
# -------------------------------
def go_to_home():
    st.session_state.page = "Home"

def go_to_detection():
    st.session_state.page = "Disease Detection"

def go_to_model_info():
    st.session_state.page = "Model Info"

def go_to_about():
    st.session_state.page = "About"

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
st.sidebar.radio(
    "Go to",
    ["Home", "Disease Detection", "Model Info", "About"],
    index=["Home", "Disease Detection", "Model Info", "About"].index(st.session_state.page),
    key="sidebar_nav",
    on_change=lambda: st.session_state.update({"page": st.session_state.sidebar_nav})
)

# -------------------------------
# Home Page
# -------------------------------
if st.session_state.page == "Home":
    st.markdown("<div class='main-title'>Maize Disease Detection System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>An intelligent deep learning system that identifies maize leaf diseases using computer vision and AI.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Start Detection", on_click=go_to_detection):
            pass

# -------------------------------
# Disease Detection Page
# -------------------------------
elif st.session_state.page == "Disease Detection":
    st.markdown("<div class='main-title'>Maize Disease Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>Upload a maize leaf image to identify if it is healthy or affected by a disease.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

            img_resized = img.resize(IMG_SIZE)
            img_array = np.expand_dims(np.array(img_resized), axis=0)
            img_array = preprocess_input(img_array)

            with st.spinner("Analyzing image..."):
                preds = model.predict(img_array, verbose=0)

            pred_idx = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(np.max(preds) * 100)

            # Style output
            if pred_class.lower() == "healthy":
                color = "#1B4332"
                bg_color = "#E6F4EA"
                title = "Healthy"
            else:
                color = "#7C2D12"
                bg_color = "#FEF2F2"
                title = "Diseased"

            st.markdown(f"""
                <div class='result-box' style='background-color:{bg_color}; border-left: 5px solid {color};'>
                    <h3 style='color:{color}; margin-bottom:0;'>{title}</h3>
                    <p style='color:{color}; font-size:1.2em; margin-top:0.2em;'>
                        Prediction: <strong>{pred_class}</strong><br>
                        Confidence: <strong>{confidence:.2f}%</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Treatment info
            treatment_info = {
                "Healthy": "The leaf appears healthy. No treatment needed.",
                "Blight": "Blight detected. Remove infected leaves and apply a copper-based fungicide.",
                "Common Rust": "Common Rust detected. Use rust-resistant varieties and apply fungicides early.",
                "Gray_Leaf_Spot": "Gray Leaf Spot detected. Rotate crops and avoid overhead irrigation."
            }

            st.markdown("### Recommended Action")
            st.write(treatment_info.get(pred_class, "No treatment information available."))

            # Confidence chart
            st.markdown("### Prediction Confidence by Class")
            prob_data = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Confidence (%)": [p * 100 for p in preds[0]]
            })

            chart = (
                alt.Chart(prob_data)
                .mark_bar(size=40, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Class", sort=None, title=None),
                    y=alt.Y("Confidence (%)", title="Confidence (%)"),
                    color=alt.Color("Class", legend=None),
                    tooltip=["Class", "Confidence (%)"]
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)

            st.dataframe(
                prob_data.style.format({"Confidence (%)": "{:.2f}"}).background_gradient(axis=None, cmap="Greens")
            )

        except Exception as e:
            st.error("Failed to process the uploaded image. Please try again with a valid file.")
            st.exception(e)
            st.stop()
    else:
        st.info("Please upload an image to begin detection.")

# -------------------------------
# Model Info Page
# -------------------------------
elif st.session_state.page == "Model Info":
    st.markdown("<div class='main-title'>Model Information</div>", unsafe_allow_html=True)
    st.write("""
    **Architecture:** MobileNetV2 (Transfer Learning)  
    **Input Size:** 224 × 224 × 3  
    **Optimizer:** Ngcebo  
    **Loss Function:** Categorical Crossentropy  
    **Accuracy Goal:** 80–90% on validation  
    **Framework:** TensorFlow / Keras  
    """)
    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, "r") as f:
                mapping = json.load(f)
            st.markdown("**Class Indices Mapping:**")
            st.json(mapping)
        except Exception as e:
            st.error("Failed to load class indices mapping.")
            st.exception(e)
    else:
        st.warning("Class indices mapping file not found.")

# -------------------------------
# About Page
# -------------------------------
elif st.session_state.page == "About":
    st.markdown("<div class='main-title'>About This Project</div>", unsafe_allow_html=True)
    st.write("""
    This project leverages AI to identify common maize leaf diseases 
    using image classification. Developed as a practical tool to help 
    farmers detect crop diseases early.

    **Purpose**
    - Support early detection and management of maize diseases.
    - Reduce yield loss and promote sustainable agriculture.

    **Technologies Used**
    - TensorFlow & Keras (AI model)
    - Streamlit (Web deployment)
    - Altair & Pandas (Visualization)

    **Developed by:** Ngcebo Maphumulo
    """)
