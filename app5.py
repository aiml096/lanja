import streamlit as st
import numpy as np
from PIL import Image
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tensorflow.keras.models import load_model  # type: ignore
import os

# Load pre-trained models
keras_model = load_model('C:/Users/bharg/Desktop/vs/skin_disease_classifier4.keras')  # TensorFlow/Keras model
tokenizer = GPT2Tokenizer.from_pretrained('C:/Users/bharg/OneDrive/Desktop/skin/vs/fine_tuned_gpt2')
chatbot_model = GPT2LMHeadModel.from_pretrained('C:/Users/bharg/OneDrive/Desktop/skin/vs/fine_tuned_gpt2')
pytorch_model = GPT2LMHeadModel.from_pretrained('gpt2')
pytorch_model.load_state_dict(torch.load('C:/Users/bharg/OneDrive/Desktop/skin/vs/pytorch_model2.pth'))
pytorch_model.eval()

class_names = ['BA- cellulitis','BA-impetigo', 'FU-athlete-foot','FU-nail-fungus','FU-ringworm', 
               'PA-cutaneous-larva-migrans','VI-chickenpox','VI-shingles']

def preprocess_image(image, target_size=(28, 28)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_with_keras(image):
    preprocessed_image = preprocess_image(image)
    prediction = keras_model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    return class_names[predicted_class]

def classify_with_pytorch(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        outputs = pytorch_model(image)
        predicted_class = torch.argmax(outputs, dim=1).item()
        return class_names[predicted_class]

def get_chatbot_response(question):
    inputs = tokenizer.encode(question, return_tensors='pt')
    reply_ids = chatbot_model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# Set the page configuration
st.set_page_config(
    page_title="Skin Disease Classifier & Care Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add background styling using CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
        color: #333;
    }
    .reportview-container {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2e8b57;
    }
    h2 {
        color: #4682B4;
    }
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit App Layout
st.title("Skin Disease Classifier & Care Chatbot 🤖💬")

# Tabs for better navigation
tab1, tab2 = st.tabs(["🔬 Disease Classification", "💬 Skin Care Chatbot"])

with tab1:
    st.header("Image Classification")
    
    # Option to either upload or capture an image
    uploaded_image = st.file_uploader("Upload an image for skin disease classification", type=["jpg", "jpeg", "png"])
    scanned_image = st.camera_input("Or, take a picture with your camera")
    
    image = None
    file_name = None  # Initialize variable to store file name
    if uploaded_image:
        file_name = os.path.splitext(uploaded_image.name)[0]  # Extract file name without extension
        image = Image.open(uploaded_image)
        st.image(image, caption=f"Uploaded Image: {file_name}", use_column_width=True)
    elif scanned_image:
        image = Image.open(scanned_image)
        st.image(image, caption="Scanned Image", use_column_width=True)

    if image:
        # Let user select model
        model_choice = st.radio("Select Model for Classification", ("Keras", "PyTorch"))
        
        if st.button("Classify"):
            st.write("Classifying using the selected model...")
            if model_choice == "Keras":
                predicted_name = classify_with_keras(image)
            else:
                predicted_name = classify_with_pytorch(image)
            st.success(f"Image Name: **{file_name if file_name else 'Scanned Image'}**")

with tab2:
    st.header("Skin Care Chatbot")
    
    user_question = st.text_input("Ask a question about skin care:")
    
    if st.button("Ask Chatbot"):
        if user_question:
            with st.spinner("Generating response..."):
                response = get_chatbot_response(user_question)
            st.write(f"Chatbot Response: **{response}**")
        else:
            st.warning("Please enter a question.")

# Footer for additional style
st.markdown("""
    <div class="footer">
    <hr>
    👨‍⚕️ Developed to assist with skin disease classification and provide skin care tips. Stay safe and take care of your skin!
    </div>
    """, unsafe_allow_html=True)
