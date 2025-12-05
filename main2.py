import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="PhytoLens", 
                   page_icon="🌿", 
                   layout="centered"
                   )

# LOAD MODEL
MODEL_PATH = "model/trained_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# IMAGE PREDICTION FUNCTION
def model_prediction(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((128, 128))    # match training size
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])   # convert to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# DISEASE INFO DATABASE
disease_info = {
    "Apple___Apple_scab": {
        "brief": "Dark spots appear on leaves and fruits.",
        "steps": [
            "Remove infected leaves and fruits from the tree.",
            "Do not leave fallen leaves on the ground.",
            "Spray recommended fungicide from nearby agri shop.",
            "Keep trees open so air can pass easily."
        ]
    },

    "Apple___Black_rot": {
        "brief": "Black spots on leaves and fruits that rot later.",
        "steps": [
            "Cut and throw away infected fruits.",
            "Remove dried or dead branches.",
            "Spray copper-based fungicide.",
            "Keep the area clean under the trees."
        ]
    },

    "Apple___Cedar_apple_rust": {
        "brief": "Yellow or rusty spots on leaves.",
        "steps": [
            "Cut nearby juniper/cedar trees if possible.",
            "Spray fungicide in early season.",
            "Remove infected leaves from the ground."
        ]
    },

    "Apple___healthy": {
        "brief": "Your apple leaf is healthy.",
        "steps": ["No action needed."]
    },

    "Blueberry___healthy": {
        "brief": "The plant is healthy.",
        "steps": ["No action needed."]
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "brief": "White powder appears on leaves.",
        "steps": [
            "Cut very infected leaves.",
            "Do not water from top; water near roots.",
            "Spray sulfur or baking-soda based spray.",
            "Keep plant in open air."
        ]
    },

    "Cherry_(including_sour)___healthy": {
        "brief": "The plant is healthy.",
        "steps": ["No action needed."]
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "brief": "Brown/gray spots on leaves that spread slowly.",
        "steps": [
            "Do crop rotation next season.",
            "Remove dry plant waste after harvest.",
            "Use disease-resistant seeds if possible.",
            "Spray fungicide only if disease spreads fast."
        ]
    },

    "Corn_(maize)___Common_rust_": {
        "brief": "Brown powdery bumps on leaves.",
        "steps": [
            "Use rust-resistant seeds next time.",
            "Spray fungicide if rust becomes too much.",
            "Avoid over-watering the crop."
        ]
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "brief": "Long brown patches on leaves.",
        "steps": [
            "Remove infected leaves if infection is small.",
            "Spray recommended fungicide on time.",
            "Rotate crops every year."
        ]
    },

    "Corn_(maize)___healthy": {
        "brief": "Corn leaf is healthy.",
        "steps": ["No action needed."]
    },

    "Grape___Black_rot": {
        "brief": "Black spots appear on grapes and leaves.",
        "steps": [
            "Remove infected fruits and leaves.",
            "Keep vines open for airflow.",
            "Spray fungicide from agri shop."
        ]
    },

    "Grape___Esca_(Black_Measles)": {
        "brief": "Leaves turn yellow/brown and fruits dry early.",
        "steps": [
            "Cut and throw infected branches.",
            "Do not injure the main stem while pruning.",
            "Ask local expert for suitable fungicide timing."
        ]
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "brief": "Spots on leaves that spread during rains.",
        "steps": [
            "Remove infected leaves.",
            "Spray suggested fungicides.",
            "Keep the plant open for airflow."
        ]
    },

    "Grape___healthy": {
        "brief": "Grape leaf is healthy.",
        "steps": ["No action needed."]
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "brief": "Very serious disease; fruits become small and taste bad.",
        "steps": [
            "Remove the infected tree (no cure).",
            "Control whitefly/psyllid insects.",
            "Inform nearest agriculture department."
        ]
    },

    "Peach___Bacterial_spot": {
        "brief": "Small dark spots on leaves and fruits.",
        "steps": [
            "Remove heavily infected leaves.",
            "Spray copper-based spray.",
            "Use clean and healthy plants for planting."
        ]
    },

    "Peach___healthy": {
        "brief": "Healthy leaf.",
        "steps": ["No action needed."]
    },

    "Pepper,_bell___Bacterial_spot": {
        "brief": "Wet-looking small spots on pepper leaves.",
        "steps": [
            "Remove very infected plants.",
            "Spray copper bactericide.",
            "Rotate crops and clean tools."
        ]
    },

    "Pepper,_bell___healthy": {
        "brief": "Plant is healthy.",
        "steps": ["No action needed."]
    },

    "Potato___Early_blight": {
        "brief": "Brown spots with rings on potato leaves.",
        "steps": [
            "Remove lower infected leaves.",
            "Spray suitable fungicide.",
            "Do crop rotation next year."
        ]
    },

    "Potato___Late_blight": {
        "brief": "Leaves turn dark and die very fast.",
        "steps": [
            "Remove and destroy infected plants immediately.",
            "Spray copper fungicide.",
            "Avoid watering leaves in evening."
        ]
    },

    "Potato___healthy": {
        "brief": "Healthy potato leaf.",
        "steps": ["No action needed."]
    },

    "Raspberry___healthy": {
        "brief": "Healthy leaf.",
        "steps": ["No action needed."]
    },

    "Soybean___healthy": {
        "brief": "Healthy leaf.",
        "steps": ["No action needed."]
    },

    "Squash___Powdery_mildew": {
        "brief": "White powder appears on leaf surface.",
        "steps": [
            "Cut infected leaves.",
            "Spray potassium bicarbonate or sulfur.",
            "Increase airflow around plants."
        ]
    },

    "Strawberry___Leaf_scorch": {
        "brief": "Leaves turn brown from edges.",
        "steps": [
            "Remove dried leaves.",
            "Use recommended fungicide.",
            "Avoid overhead watering."
        ]
    },

    "Strawberry___healthy": {
        "brief": "Healthy leaf.",
        "steps": ["No action needed."]
    },

    "Tomato___Bacterial_spot": {
        "brief": "Brown water-soaked spots on leaves.",
        "steps": [
            "Remove infected leaves.",
            "Spray copper spray.",
            "Clean tools and avoid touching wet plants."
        ]
    },

    "Tomato___Early_blight": {
        "brief": "Brown spots with rings on lower leaves.",
        "steps": [
            "Remove bottom infected leaves.",
            "Spray copper/chlorothalonil fungicide.",
            "Do crop rotation yearly."
        ]
    },

    "Tomato___Late_blight": {
        "brief": "Very fast spreading disease; leaves turn black.",
        "steps": [
            "Remove infected plants immediately.",
            "Spray copper fungicide during outbreaks.",
            "Keep plant area dry and airy."
        ]
    },

    "Tomato___Leaf_Mold": {
        "brief": "Yellow patches on top, mold under leaf.",
        "steps": [
            "Increase airflow.",
            "Remove infected leaves.",
            "Spray suitable fungicide."
        ]
    },

    "Tomato___Septoria_leaf_spot": {
        "brief": "Small brown dots on leaves.",
        "steps": [
            "Remove infected leaves.",
            "Spray fungicide (follow label).",
            "Keep leaves dry, water at base."
        ]
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "brief": "Tiny insects make leaves yellow with webbing.",
        "steps": [
            "Spray neem oil or soap solution.",
            "Wash leaves with water gently.",
            "Keep plants cool and well-watered."
        ]
    },

    "Tomato___Target_Spot": {
        "brief": "Dark round spots with rings on leaves.",
        "steps": [
            "Remove infected leaves.",
            "Spray suitable fungicide.",
            "Keep plant area clean."
        ]
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "brief": "Leaves curl upward; plant becomes small.",
        "steps": [
            "Remove infected plants (no cure).",
            "Control whiteflies with sprays.",
            "Use resistant seeds if available."
        ]
    },

    "Tomato___Tomato_mosaic_virus": {
        "brief": "Light and dark patches on leaves.",
        "steps": [
            "Remove infected plants.",
            "Wash hands and tools before touching plants.",
            "Use disease-free seeds."
        ]
    },

    "Tomato___healthy": {
        "brief": "Tomato plant is healthy.",
        "steps": ["No action needed."]
    }
}

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    st.markdown("""
    ## 🌱 PhytoLens

    A smart system for rapid and accurate plant disease detection.  
    Upload a leaf image to analyze plant health using advanced AI models.

    ---

    ### How It Works
    1. **Upload:** Select a leaf image on the **Disease Detection** page.  
    2. **Analyze:** The system processes the image using deep learning.  
    3. **Result:** View detected disease and recommended actions.

    ---

    ### Why PhytoLens?
    - High detection accuracy  
    - Fast processing  
    - Simple and intuitive interface  

    ---

    Navigate to **Disease Detection** to begin.
""")

st.title("PhytoLens-Plant Disease Detector")
file = st.file_uploader("Upload a plant leaf image")
if(st.button("Show Image")):
        st.image(file,width='stretch')

if st.button("Predict") and file:
    result_index = model_prediction(file)
    class_name = class_names[result_index]

    # After computing class_name and confidence...
    st.success(f"Predicted Disease: **{class_name}**")

    # show brief summary
    info = disease_info.get(class_name)
    if info:
        st.markdown("### 📝 Brief")
        st.info(info["brief"])

        with st.expander("📋 Step-by-step treatment (Easy Guide)"):
            for i, step in enumerate(info["steps"], start=1):
                st.markdown(f"**{i}. {step}**")
    else:
        st.warning("No information available for this disease.")

