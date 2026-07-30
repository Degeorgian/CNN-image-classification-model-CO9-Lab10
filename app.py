import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Acne vs Eczema Classifier", page_icon="🩺")

# --- App Title and Description ---
st.title("Acne vs Eczema Classifier")
st.write("""
Upload a clear photo of the skin condition. 
The model will predict whether it is **Acne** or **Eczema** and provide a confidence score.
""")

# --- Load the Model ---
# Using @st.cache_resource ensures the model is loaded only once, saving time on rerun
@st.cache_resource
def load_model():
    # Replace 'best_model.keras' with the exact filename of your saved model
    return tf.keras.models.load_model('./Dataset/best_model.keras')

model = load_model()

# --- Image Upload Section ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # --- Processing the User Input ---
    st.write("Processing image and running prediction...")
    
    # 1. Ensure the image is in RGB format (removes alpha channels if present)
    image = image.convert('RGB')
    
    # 2. Resize to the dimensions the model was trained on (224x224)
    image = image.resize((224, 224))
    
    # 3. Convert PIL image to a numpy array
    img_array = np.array(image)
    
    # 4. Scale the pixel values exactly like the training data (0 to 1)
    img_array = img_array / 255.0
    
    # 5. Expand dimensions to create a batch of 1 (shape becomes: 1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # --- Prediction Button & Logic ---
    if st.button('Predict'):
        # Get the prediction probability (since we used Sigmoid, this is a single value)
        prediction = model.predict(img_batch)
        probability = prediction[0][0]
        
        # --- Logic and Display Results ---
        # > 0.5 is Eczema (1), <= 0.5 is Acne (0)
        if probability > 0.5:
            predicted_class = "Eczema"
            # Calculate confidence score as a percentage
            confidence = probability * 100
            st.error(f"**Prediction:** {predicted_class}")
        else:
            predicted_class = "Acne"
            # Since <=0.5 is Acne, the confidence is (1 - probability) * 100
            confidence = (1 - probability) * 100
            st.warning(f"**Prediction:** {predicted_class}")
            
        st.info(f"**Confidence Score:** {confidence:.2f}%")