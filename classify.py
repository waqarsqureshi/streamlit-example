import streamlit as st
import timm, os
import torch
import cv2
from PIL import Image, ImageDraw, Image,ImageFont
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from timm.models import create_model, load_checkpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pathlib import Path
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from datetime import datetime
import zipfile  # Importing zipfile module
import time
import glob
#=========================

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, HiResCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image, create_labels_legend
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam import DeepFeatureFactorization, run_dff_on_image
from pytorch_grad_cam.utils.image import show_factorization_on_image

#===========================
# Constants
class_no = 10
class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
color_mapping = {
    '1': 'red', '2': 'red',
    '3': 'orange', '4': 'orange',
    '5': 'lightblue', '6': 'lightblue',
    '7': 'blue', '8': 'blue',
    '9': 'green', '10': 'green'
}

methods = \
        {"gradcam": GradCAM,
         "gradcam++": GradCAMPlusPlus,
         "xgradcam": XGradCAM,
         "layercam": LayerCAM,
         "hirescam": HiResCAM}

NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
SIZE = 384
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'


#==========================================================================
#-----------------
def get_transform(size,NORMALIZE_MEAN,NORMALIZE_STD):
    if size ==SIZE:
        transforms = [T.Resize((SIZE,SIZE), interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]

        transforms = T.Compose(transforms)
        return transforms
    return 1
#-----------------    
def reshape_transform(tensor, height=12, width=12):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result
#-----------------

#-----------------
def create_load_model(name,checkpoint_path):
    
    model = create_model(
        name,
        pretrained=False, 
        num_classes=class_no
        )

    load_checkpoint(model,checkpoint_path)
    return model
#-----------------

from PIL import ImageFont

def add_classes_to_image(image, top3_predictions):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Load the default font
    # Scale the font to the desired size
    try:
        font = font.font_variant(size=20)
    except AttributeError:
        # If the default font doesn't support font_variant, you may need to find a specific TrueType font file
        pass

    y_position = 10
    x_position = 10
    for i, (class_name, prob) in enumerate(top3_predictions):
        text = f"Prediction {i + 1}: {class_name} ({prob})"
        color = color_mapping[class_name]  # Get the color based on the class index
        draw.text((10, y_position), text, font=font, fill=color)
        y_position += 20

    return image

#==========================================================================

def classify_image(image_path):
    use_cuda = False
    if torch.cuda.is_available():
        device = 'cuda'
        use_cuda = True
    else:
        device = 'cpu'
        use_cuda = False
    
    model = create_load_model('swinv2_base_window12to24_192to384.ms_in22k_ft_in1k', checkpoint_path_swin)
    model.to(device)
    model.eval()
    transform = get_transform(SIZE, NORMALIZE_MEAN, NORMALIZE_STD)
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = torch.softmax(model(img_tensor), dim=1)
        
    top3_prob, top3_indices = torch.topk(output, k=3)
    top3_labels = [class_names[int(idx)] for idx in top3_indices[0]]
    top3_probs = (top3_prob[0] * 100).tolist()
    top3_predictions = [(label, f"{prob:.2f}%") for label, prob in zip(top3_labels, top3_probs)]
    predicted_class_name = top3_labels[0]
    image_with_classes = add_classes_to_image(img, top3_predictions)
    
    return predicted_class_name, image_with_classes, top3_predictions 


def classify(folder_path):
        images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjP][npP][gG]*'), recursive=True)]
        # Sort the image paths
        images.sort()
        if folder_path:                
            if images:
                    # set the states
                    image_index = st.session_state.get("image_index", 0)
                    predictions = st.session_state.get("predictions", [])
                    indices = st.session_state.get("indices", [])
                    results = st.session_state.get("results", [])
                    # Custom graphical buttons
                    left_column, right_column = st.columns(2)
                    if left_column.button("Previous"):
                        image_index = (image_index - 1) % len(images)
                        image_path = os.path.join(folder_path, images[image_index])

                    if right_column.button("Next"):
                        image_index = (image_index + 1) % len(images)
                        image_path = os.path.join(folder_path, images[image_index])

                    # process the image
                    image_path = os.path.join(folder_path, images[image_index])
                    image_names = os.path.basename(images[image_index])  # Extracting only the image names
                    predicted_class_name, image_with_classes, top3_predictions = classify_image(image_path)
                    st.image(image_with_classes, caption=image_names)
                    predictions.append(predicted_class_name)
                    indices.append(image_index)
                    results.append((images[image_index], top3_predictions))
                    # save the states    
                    st.session_state.image_index = image_index
                    st.session_state.predictions = predictions
                    st.session_state.indices = indices
                    st.session_state.results = results
                    st.write(f"Image Index: {image_index}")  
                    # Get the sorted indices based on the predictions and Sort the predictions and indices based on the sorted indices
                    sorted_indices = [i for i, _ in sorted(enumerate(st.session_state.predictions), key=lambda x: x[1])]
                    sorted_predictions = [st.session_state.predictions[i] for i in sorted_indices]
                    sorted_image_indices = [st.session_state.indices[i] for i in sorted_indices]
                    # Display the current prediction class
                

                    #Plot the graph
                    #plt.scatter(st.session_state.indices, st.session_state.predictions, marker='o')
                    plt.scatter(sorted_image_indices, sorted_predictions, marker='o')
                    plt.xlabel('Image Index')
                    plt.ylabel('Prediction')
                    plt.xticks(indices)
                    plt.title('Predictions Over Time')
                    plt.grid(True)  # Add grid lines

                    col1, col2, col3 = st.columns(3)

                    # Display the plot in the first column
                    col1.pyplot(plt)

                    # Display the current prediction as a colored circle button with the class name inside in the second column
                    color = color_mapping[predicted_class_name]
                    circle_button_html = f"""<div style="width: 60px; height: 60px; background-color: {color}; border-radius: 50%; text-align: center; line-height: 60px; font-size: 20px; color: white;">{predicted_class_name}</div>"""
                    col2.markdown(circle_button_html, unsafe_allow_html=True)
                
                    # Save to CSV button
                    if col3.button("Save Results to CSV"):
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        csv_filename = f"{timestamp}_results.csv"
                        with open(csv_filename, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(["Image Name", "Prediction 1", "Probability 1", "Prediction 2", "Probability 2", "Prediction 3", "Probability 3"])
                            for result in results:
                                image_name, top3 = result
                                row = [image_name] + [item for sublist in top3 for item in sublist]
                                writer.writerow(row)
                        col3.success(f"Results saved to {csv_filename}")

            else:
                st.warning("No images found in the folder.")
        else:
            st.error("The zip is not valid and is None")



def gradCam(folder_path):
    threshold = st.slider("Select a threshold value:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    if folder_path:
        if os.path.isdir(folder_path):
            images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjP][npP][gG]*'), recursive=True)]
            # Sort the image paths
            images.sort()
            image_names = [os.path.basename(image) for image in images]  # Extracting only the image names
            selected_image_name = st.selectbox("Choose an image:", image_names)
            selected_image_index = image_names.index(selected_image_name)
            image_path = images[selected_image_index]  # Get the full path of the selected image
            image_names = os.path.basename(images[selected_image_index])

            rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
            #rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (384, 384))
            rgb_img_float = np.float32(rgb_img) / 255
            img_tensor = preprocess_image(rgb_img_float, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            use_cuda = False
            if torch.cuda.is_available():
                device = 'cuda'
                use_cuda = True
            else:
                device = 'cpu'
                use_cuda = False

            model = create_load_model('swinv2_base_window12to24_192to384.ms_in22k_ft_in1k', checkpoint_path_swin)
            model.to(device)
            model.eval()
            targets = None
            # Select method
            method = st.selectbox("Select CAM method:", list(methods.keys()))

            # select layer
            target_layers = [model.layers[-1].blocks[-1].norm2]
            # use the model in cuda
            cam = methods[method](model=model,
                    target_layers=target_layers,
                    use_cuda=use_cuda,
                    reshape_transform=reshape_transform)

            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            mask = (grayscale_cam > threshold).astype(np.float32)  # Apply the user-defined threshold
            mask_3_channel = np.stack([mask] * 3, axis=-1)
            #mask_image = (rgb_img * mask_3_channel).astype(np.uint8)
            cam_image = show_cam_on_image(rgb_img_float, mask, use_rgb=False, image_weight=0.8, colormap=cv2.COLORMAP_HOT)
            cam_image = cv2.resize(cam_image, (700, 300))
            # Display results
            st.image(cam_image, caption=image_names, use_column_width=True)
        else:
            st.warning("No images found in the folder.")
    else:
        st.error("The path is not a valid directory.")

                


if __name__ == "__main__":
    # Display the logo
    # Create two columns
    options = ["Image Classification", "GRAD-CAM"]
    selected_option = st.sidebar.selectbox("Choose an example:", options)
    left_column, right_column = st.columns(2)
    logo_path1 = "/home/pms/Pictures//TU Dublin Logo_resized.jpg"
    left_column.image(logo_path1, width=100)
    logo_path2 = "/home/pms/Pictures//pms.png"
    right_column.image(logo_path2, width=100)
    st.title("Pavement Surface Condition Index")

    uploaded_zip = st.file_uploader("Upload a ZIP file of images:", type=['zip'])

    if uploaded_zip is not None:
        temp_dir_path = "/home/pms/streamlit-example/temp/"
        os.makedirs(temp_dir_path, exist_ok=True)

        # Check if the unique_temp_dir is already in the session state
        if 'unique_temp_dir' not in st.session_state:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get the current date and time
            unique_temp_dir = os.path.join(temp_dir_path, timestamp)  # Use the timestamp as the directory name
            os.makedirs(unique_temp_dir)
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(unique_temp_dir)
                st.session_state.unique_temp_dir = unique_temp_dir  # Store the directory path in the session state
            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP file. Please upload a valid ZIP file.")

        images = [f for f in glob.glob(os.path.join(st.session_state.unique_temp_dir, '**', '*.[pjP][npP][gG]*'), recursive=True)]
        if images:
            if selected_option == "Image Classification":
                st.title("Image Classification")
                classify(st.session_state.unique_temp_dir)

            elif selected_option == "GRAD-CAM":
                st.title("Welcome to GRAD-CAM")
                gradCam(st.session_state.unique_temp_dir)
        else:
            st.error("No images found in the ZIP file. Please upload a valid ZIP file containing images.")
    else:
        st.warning("Please upload a ZIP file containing images.")


