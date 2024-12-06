import streamlit as st  
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import keras
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import warnings
warnings.filterwarnings("ignore")

breed_classes = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih', 'Blenheim spaniel', 'papillon', 'toy terrier',
 'Rhodesian ridgeback', 'Afghan hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black', 'Walker hound',
 'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound',
 'Norwegian elkhound', 'otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier',
 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier',
 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire', 'Lakeland terrier', 'Sealyham terrier', 'Airedale',
 'cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'miniature schnauzer', 'giant schnauzer',
 'standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft', 'West Highland white terrier',
 'Lhasa', 'flat', 'curly', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short',
 'vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'clumber', 'English springer',
 'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke',
 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie',
 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher',
 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff',
 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky',
 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'chow',
 'keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle', 'miniature poodle', 'standard poodle',
 'Mexican hairless', 'dingo', 'dhole', 'African hunting dog']

class_names = ['alert', 'angry', 'frown', 'happy', 'relax']

st.set_page_config(
    page_title="Dog Emotion & Breed Detection",
    page_icon=":dog:",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Custom CSS for a minimal, elegant look
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root {
    --accent-color: #E59866; /* Warm accent color */
    --text-color: #333333;
    --bg-gradient-start: #FAF9F7;
    --bg-gradient-end: #F2ECE3;
    --card-bg: #FFFFFF;
    --info-bg: #F9F9F9;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(to bottom, var(--bg-gradient-start), var(--bg-gradient-end));
    color: var(--text-color);
}

.block-container {
    max-width: 800px;
    margin: auto;
    padding: 2rem;
}

.reportview-container .main .block-container {
    padding: 0;
}

.header-container {
    text-align: center;
    margin-bottom: 2rem;
}

.header-container h1 {
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: var(--text-color);
}

.header-container h3 {
    font-weight: 400;
    color: #555;
    margin-top: 0;
}

.upload-container, .results-container {
    background: var(--card-bg);
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 2rem;
}

.sidebar .sidebar-content {
    background: #FFFFFF;
    border-right: 1px solid #DDD;
    color: var(--text-color);
}

.sidebar .sidebar-content h2 {
    text-align: center;
    color: var(--accent-color);
    margin-bottom: 1rem;
}

.sidebar .sidebar-content p {
    text-align: center;
    font-weight: 400;
    font-size: 0.95rem;
    color: #555;
}

.stButton>button {
    background: var(--accent-color);
    color: #fff;
    border: none;
    border-radius: 5px;
    font-weight: 600;
    padding: 0.6rem 1rem;
    transition: background 0.3s;
}

.stButton>button:hover {
    background: #d58e5c;
}

.results-container h2 {
    margin-bottom: 1rem;
    color: var(--text-color);
    font-weight: 600;
}

.stImage img {
    border-radius: 5px;
}

.info-box {
    background: var(--info-bg);
    padding: 1rem;
    border-radius: 5px;
    margin-top: 1rem;
    color: #555;
    font-size: 0.95rem;
}

.info-box strong {
    color: var(--accent-color);
}

/* Custom badges for messages */
.stAlert, .stError, .stInfo, .stSuccess, .stWarning {
    border-radius: 5px;
    color: #333 !important;
    font-weight: 600;
}

.stSuccess {
    background: #E9F7EF;
}

.stWarning {
    background: #FEF9E7;
}

.stError {
    background: #FDEDEC;
}

.stInfo {
    background: #EBF5FB;
}

/* Hide Streamlit default footer */
footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🐶 Dog Emotion & Breed Classifier")
    st.write("Find out your dog's emotion and breed with a single image upload.")

st.markdown("<div class='header-container'><h1>Dog Emotion & Breed Detection</h1><h3>Upload an image of your dog and let AI do the rest.</h3></div>", unsafe_allow_html=True)

file = st.file_uploader("", type=["jpg", "png"])

def preprocess_for_prediction(image_data, target_size, incep):
    image = ImageOps.fit(image_data, target_size)
    img = np.asarray(image)
    if incep:
        img = preprocess_input(img)
    img_reshape = img[np.newaxis, ...]
    return img_reshape

def import_and_predict(image_data, emotion_model, breed_model):
    # Emotion model (224x224)
    emotion_input = preprocess_for_prediction(image_data, (224, 224), False)
    emotion_prediction = emotion_model.predict(emotion_input)
    # Breed model (299x299)
    breed_input = preprocess_for_prediction(image_data, (299, 299), True)
    breed_prediction = breed_model.predict(breed_input)
    return emotion_prediction, breed_prediction

if file is None:
    st.markdown("<div class='upload-container'><p>Please upload an image file to proceed.</p></div>", unsafe_allow_html=True)
else:
    # Load models
    emotion_model = keras.models.load_model('emotion_model.keras')
    breed_model = keras.models.load_model('breed_model.keras')
    
    image = Image.open(file)
    st.markdown("<div class='upload-container'>", unsafe_allow_html=True)
    st.image(image, use_column_width=True, caption="Uploaded Image")
    st.markdown("</div>", unsafe_allow_html=True)

    emotion_preds, breed_preds = import_and_predict(image, emotion_model, breed_model)

    emotion_confidence = np.max(emotion_preds) * 100
    detected_emotion = class_names[np.argmax(emotion_preds)]

    breed_confidence = np.max(breed_preds) * 100
    detected_breed_index = breed_classes[np.argmax(breed_preds)]

    st.markdown("<div class='results-container'>", unsafe_allow_html=True)
    st.markdown("<h2>Results</h2>", unsafe_allow_html=True)
    st.write(f"**Emotion:** {detected_emotion} ({emotion_confidence:.2f}% confidence)")
    st.write(f"**Breed:** {detected_breed_index} ({breed_confidence:.2f}% confidence)")

    # Display insights
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Emotion Insights")
        if detected_emotion == "happy":
            st.success("Your dog seems **happy!** A joyful companion is always a treat.")
        elif detected_emotion == "alert":
            st.warning("Your dog is **alert.** Something's caught their attention.")
        elif detected_emotion == "angry":
            st.error("Your dog may be **angry** or stressed. Consider a soothing approach.")
        elif detected_emotion == "frown":
            st.warning("Your dog may be feeling **uneasy**. A bit more care could help.")
        elif detected_emotion == "relax":
            st.info("Your dog appears **relaxed**. All seems calm and comfortable.")

    with col2:
        st.markdown("### Breed Details")
        st.info(f"**{detected_breed_index}**")
        st.markdown("<div class='info-box'><p>Knowing your dog's breed can guide you in offering proper exercise, diet, and grooming. Each breed has unique traits and needs, so consider researching more about <strong>" + detected_breed_index + "</strong> to ensure the best care.</p></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
