# Importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# Hide depreciation warnings which directly don't affect the working of the application
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


# Set some pre-defined configurations for the page
st.set_page_config(
    page_title="Dog Emotion Detection",
    page_icon=":dog:",
    initial_sidebar_state="auto"
)

# Hide the part of the code for adding custom CSS styling
hide_streamlit_style = """
	<style>
    #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to predict the class of the images based on the model results
def prediction_cls(prediction): 
    for key, clss in enumerate(class_names): 
        if np.argmax(prediction) == clss:
            return key

with st.sidebar:
    #st.image('dog_emotion_logo.png')  # Replace with your logo image
    st.title("Dog Emotion Classifier")
    st.subheader("Accurately detect the emotion and breed of your dog using AI.")

st.write("""
         # Dog Emotion Detection
         Upload an image of your dog to detect its breed and current emotion.
         """)

file = st.file_uploader("", type=["jpg", "png"])
# Function to preprocess and predict
def preprocess_for_prediction(image_data, target_size, incep):
    image = ImageOps.fit(image_data, target_size)
    img = np.asarray(image)
    if incep:
        img = preprocess_input(img)
    img_reshape = img[np.newaxis, ...]
    return img_reshape

def import_and_predict(image_data, emotion_model, breed_model):
    # Process for emotion model (224x224)
    emotion_input = preprocess_for_prediction(image_data, (224, 224), False)
    emotion_prediction = emotion_model.predict(emotion_input)
    
    # Process for breed model (299x299)
    breed_input = preprocess_for_prediction(image_data, (299, 299), True)
    breed_prediction = breed_model.predict(breed_input)
    
    return emotion_prediction, breed_prediction

# Define the emotion class names
class_names = ['alert', 'angry', 'frown', 'happy', 'relax']

if file is None:
    st.text("Please upload an image file")
else:
    # Load both models
    emotion_model = keras.models.load_model('dog_emotion_model.keras')
    breed_model = keras.models.load_model('breed_model.keras')
    
    image = Image.open(file)
    st.image(image, use_column_width=True)

    # Get both predictions
    emotion_predictions, breed_predictions = import_and_predict(image, emotion_model, breed_model)
    
    # Process emotion prediction
    emotion_confidence = np.max(emotion_predictions) * 100
    detected_emotion = class_names[np.argmax(emotion_predictions)]
    
    # Process breed prediction
    breed_confidence = np.max(breed_predictions) * 100
    detected_breed_index = breed_classes[np.argmax(breed_predictions)]

    # Display results in sidebar
    st.sidebar.markdown("## Detection Results")
    st.sidebar.success(f"Detected Emotion: {detected_emotion}")
    st.sidebar.info(f"Emotion Confidence: {emotion_confidence:.2f}%")
    st.sidebar.success(f"Detected Breed Class: {detected_breed_index}")
    st.sidebar.info(f"Breed Confidence: {breed_confidence:.2f}%")

    # Display main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## Emotion Insights")
        if detected_emotion == "happy":
            st.balloons()
            st.success("Your dog seems happy! Keep it that way!")
        elif detected_emotion == "alert":
            st.warning("Your dog is on high alert. Watch for potential stressors.")
        elif detected_emotion == "angry":
            st.error("Your dog might be angry. Try to calm them down.")
        elif detected_emotion == "frown":
            st.warning("Your dog might be sad or worried. Give them some love!")
        elif detected_emotion == "relax":
            st.info("Your dog is relaxed. Everything seems fine!")

    with col2:
        st.markdown("## Breed Detection")
        st.success(f"Detected Breed Class: {detected_breed_index}")
        st.info(f"Breed Confidence: {breed_confidence:.2f}%")