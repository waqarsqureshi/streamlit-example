import streamlit as st
import timm
import torch
from PIL import Image
import torchvision.transforms as transforms

# Function to predict the class of an image
def predict_class(image):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Load a pre-trained model
    model_name = 'resnet34'
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the predicted class
    predicted_class_idx = torch.argmax(probabilities).item()
    class_names = timm.data.resolve_data_config({}, model=model)['class_names']
    predicted_class_name = class_names[predicted_class_idx]

    return predicted_class_name

# Streamlit app
st.title("Image Classifier using timm")
uploaded_file = st.file_uploader("Choose an image from your directory", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    class_name = predict_class(image)
    st.write(f"The predicted class of the image is: {class_name}")
