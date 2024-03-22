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
import pandas as pd
import numpy as np
import time
import csv
import time
from datetime import datetime
import glob
# import for image segmentation
from util import get_palette, get_classes
from mmseg.apis import inference_model, init_model
import zipfile
import shutil
#=========================
from PIL import ImageFont

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
def create_load_model(name,checkpoint_path):
    
    model = create_model(
        name,
        pretrained=False, 
        num_classes=class_no
        )

    load_checkpoint(model,checkpoint_path)
    return model
#-----------------
#==========================================================
def add_classes_to_image(image, top3_predictions):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Load the default font

    # Scale the font to the desired size
    try:
        font = font.font_variant(size=20)
    except AttributeError:
        # If the default font doesn't support font_variant, you may need to find a specific TrueType font file
        pass

    # Get the top prediction
    top_prediction = top3_predictions[0]
    class_name, prob = top_prediction
    x_position = 10
    y_position = 10
    circle_radius = 50
    circle_center = (x_position + circle_radius, y_position + circle_radius)

    # Draw a circle with the color code of the image class
    color = color_mapping[class_name]
    draw.ellipse([x_position, y_position, x_position + 2*circle_radius, y_position + 2*circle_radius], fill=color)

    # Display the class name and its probability inside the circle
    text = f"{class_name} ({prob*100:.2f}%)"
    text_width, text_height = draw.textsize(text, font=font)
    text_position = (circle_center[0] - text_width/2, circle_center[1] - text_height/2)
    draw.text(text_position, text, font=font, fill="white")

    return image

#==========================================================
def classify_image(image_path, model, device):
    transform = get_transform(SIZE, NORMALIZE_MEAN, NORMALIZE_STD)
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = torch.softmax(model(img_tensor), dim=1)
        
    top3_prob, top3_indices = torch.topk(output, k=3)
    top3_labels = [class_names[int(idx)] for idx in top3_indices[0]]
    top3_probs = top3_prob[0].tolist()  # Save probabilities as floats
    top3_predictions = [(label, prob) for label, prob in zip(top3_labels, top3_probs)]  # Store label and float probability
    predicted_class_name = top3_labels[0]
    return predicted_class_name, top3_predictions 
#============================================================
#This function is to process image 1 by 1
def classify2(folder_path, model, device):
        images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjPJ][npP][gG]*'), recursive=True)]
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
                    predicted_class_name, image_with_classes, top3_predictions = classify_image(image_path,model,device)
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

                    # Display the plot in the first column
                    st.pyplot(plt)


                
                    # Save to CSV button
                    if st.button("Save Results to CSV"):
                        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                        csv_filename = f"{timestamp}_results.csv"
                        with open(csv_filename, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(["Image Name", "Prediction 1", "Probability 1", "Prediction 2", "Probability 2", "Prediction 3", "Probability 3"])
                            for result in results:
                                image_name, top3 = result
                                row = [image_name] + [item for sublist in top3 for item in sublist]
                                writer.writerow(row)
                        st.success(f"Results saved to {csv_filename}")

            else:
                st.warning("No images found in the folder.")
        else:
            st.error("The zip is not valid and is None")
#============================================================
def process_single_image(image_path, model, device):
    predicted_class_name, top3_predictions = classify_image(image_path, model, device)
    return predicted_class_name, top3_predictions
#============================================================
def update_progress_bar(progress_bar, current_index, total_images):
    progress_percentage = current_index / total_images
    progress_bar.progress(progress_percentage)
#============================================================
def display_results(images, folder_path):
    # Dropdown to select an image
    image_names = [os.path.basename(image) for image in images]  # Extracting only the image names

    selected_image_name = st.selectbox("Choose an image:", image_names)
    selected_image_index = image_names.index(selected_image_name)       
    
    #selected_image_index = st.selectbox("Select an image to view its result:", list(range(len(images))))
    
    if selected_image_index in st.session_state.indices:
        idx = st.session_state.indices.index(selected_image_index)
        image_name, top3_predictions = st.session_state.results[idx]
        
        # Load the image and add classes to it
        image_path = os.path.join(folder_path, images[idx])
        image = Image.open(image_path)
        image_with_classes = add_classes_to_image(image, top3_predictions)
        
        st.image(image_with_classes, caption=image_name)
    else:
        st.write("Image is not processed yet.")


#============================================================
def plot_predictions():
    sorted_indices = [i for i, _ in sorted(enumerate(st.session_state.predictions), key=lambda x: x[1])]
    sorted_predictions = [st.session_state.predictions[i] for i in sorted_indices]
    sorted_image_indices = [st.session_state.indices[i] for i in sorted_indices]    
    
    plt.scatter(sorted_image_indices, sorted_predictions, marker='o')
    plt.xlabel('Image Index')
    plt.ylabel('Prediction')
    plt.xticks(st.session_state.indices)
    plt.title('Predictions Over Time')
    plt.grid(True)
    st.pyplot(plt)
#==========================================================
def save_to_csv(folder_path):
    if st.button("Save Results to CSV"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_filename = os.path.join(folder_path, f"{timestamp}_results.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Name", "Prediction 1", "Probability 1", "Prediction 2", "Probability 2", "Prediction 3", "Probability 3"])
            for result in st.session_state.results:
                image_name, top3 = result
                row = [image_name] + [item for sublist in top3 for item in sublist]
                writer.writerow(row)
        st.success(f"Results saved to {os.path.basename(csv_filename)}")
        csv_url = f"http://192.168.1.65:8502/{os.path.basename(folder_path)}/{os.path.basename(csv_filename)}"
        st.markdown(f"[Download CSV File]({csv_url})")
#==========================================================
def classify(folder_path, model, device):
    # Check if the folder path is valid
    if folder_path:
        # Get all image paths from the folder
        images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjPJ][npP][gG]*'), recursive=True)]
        images.sort()
        # Initialize session state variables if they don't exist
        if 'image_index' not in st.session_state:
            st.session_state.image_index = 0
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'indices' not in st.session_state:
            st.session_state.indices = []
        if 'results' not in st.session_state:
            st.session_state.results = []
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False

        # Create a progress bar
        progress_bar = st.progress(0)

        # Start button to begin processing
        if st.button("Start"):
            st.session_state.is_running = True
            st.session_state.image_index = 0  # Reset the image index

        # Stop button to halt processing
        if st.button("Stop"):
            st.session_state.is_running = False

        # If the session is running and there are images left to process
        while st.session_state.is_running and st.session_state.image_index < len(images):
            image_path = os.path.join(folder_path, images[st.session_state.image_index])
            predicted_class_name, top3_predictions = process_single_image(image_path, model, device)
            
            # Update session state with the results
            st.session_state.predictions.append(predicted_class_name)
            st.session_state.indices.append(st.session_state.image_index)
            st.session_state.results.append((os.path.basename(image_path), top3_predictions))
            
            # Move to the next image
            st.session_state.image_index += 1

            # Update the progress bar
            update_progress_bar(progress_bar, st.session_state.image_index, len(images))

        # If all images have been processed or processing is stopped, display the results
        # If all images are processed or processing is stopped, display the results
        if not st.session_state.is_running or st.session_state.image_index >= len(images):
            if st.session_state.image_index >= len(images):
                st.write("Processing completed!")
            display_results(images, folder_path)
            plot_predictions()
            save_to_csv(folder_path)
    else:
        st.error("The folder path is not valid.")

#============================================================
def gradCam(folder_path, model, device):
    threshold = st.slider("Select a threshold value:", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    if folder_path:
        if os.path.isdir(folder_path):
            images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjPJ][npP][gG]*'), recursive=True)]
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
            if device == 'cuda':
                use_cuda = True

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
#=============================================================
def initialize_model(model_name, checkpointPath, config=None):
    """
    Initialize and load the model based on the provided configuration and checkpoint.
    """
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    
    if config:
        model = init_model(config, checkpointPath, device=device)
    else:
        model = create_load_model(model_name, checkpointPath)
    
    model.to(device)
    model.eval()
    
    return model, device
#==============================================================
#The function takes an path by cv2 imread and output the 576x720 image
def readImage_576x720(path):
    image = cv2.imread(path) # Read the image using opencv format
    if image.shape[:2] == (576, 720):
        orig = np.zeros((576, 720, 3), np.uint8)
        orig = image
    elif image.shape[:2] == (542,720):
        orig = np.zeros((576, 720, 3), np.uint8)
        orig [34:576,0:720] = image
    elif image.shape[:2] == (558,720):
        orig = np.zeros((576, 720, 3), np.uint8)
        orig [18:576,0:720] = image
    elif image.shape[:2] == (632,720):
        orig = np.zeros((576, 720, 3), np.uint8)
        orig  = image [56:632,0:720]
    elif image.shape[:2] == (1080,1920):
        orig = cv2.resize(image,(720,576))
    else:
        orig = cv2.resize(image,(720, 576)) #(w,h) always remember cv2 takes width and height and numpy use height and width
    return orig
#==========================================================
def result_segImage(result,palette=None,CLASS_NAMES=None):
        """compute`resulted mask` for the `img`.

        """
        seg = result.cpu()
        #the following if statement should never be executed.
        if palette is None:
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3))
            np.random.set_state(state)

        palette = np.array(palette)
        assert palette.shape[0] == len(CLASS_NAMES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            if(label == 0):
                continue
            if(label == 1):
                continue
            if(label == 2):
                continue
            if(label == 3): # road
                color_seg[seg == label, :] = [255,255,255] #color
            if(label == 4): # 
                continue
            if(label == 5): #
                continue
            if(label == 6): #
                continue
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        return color_seg
#==============================================================
def process_image(image_path, model):
    orig = readImage_576x720(image_path)
    result = inference_model(model, orig)
    result2 = result.pred_sem_seg.data[0]
    mask = result_segImage(result2, palette=get_palette('roadsurvey'), CLASS_NAMES=get_classes('roadsurvey'))
    segImage = cv2.bitwise_and(orig, mask)
    return segImage, orig
#=============================================================
def should_discard_image(segImage, orig, threshold_percentage):
    gray_mask = cv2.cvtColor(segImage, cv2.COLOR_BGR2GRAY)
    count = cv2.countNonZero(gray_mask)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thres = (orig.shape[0] * orig.shape[1]) * threshold_percentage
    return count < thres or len(contours) > 25
#=============================================================

def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))
                
#============================================================
def process_all_images(folder_path, model, seg_save_folder, crop_save_folder, threshold_percentage):
    """
    Process all images in the given folder using the provided model.
    
    Parameters:
    - folder_path: Path to the folder containing images to be processed.
    - model: The trained model used for image processing.
    - seg_save_folder: Folder where segmented images will be saved.
    - crop_save_folder: Folder where cropped images will be saved.
    - threshold_percentage: Threshold value for image processing.
    
    Returns:
    - URLs for downloading zipped segmented and cropped images.
    """
    
    # Fetch all image paths from the folder
    images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjPJ][npP][gG]*'), recursive=True)]
    total_images = len(images)
    
    # Create a progress bar in Streamlit for user feedback
    progress_bar = st.progress(0)
    
    # Process each image
    while st.session_state.current_image_index < total_images:
        image_path = images[st.session_state.current_image_index]
        
        # Allow user to stop the processing
        if st.session_state.stop_processing:
            break
        
        # Process the current image using the model
        segImage, orig = process_image(image_path, model)
        if not should_discard_image(segImage, orig, threshold_percentage):
            segImg_path = os.path.join(seg_save_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}{os.path.splitext(image_path)[1]}")
            cropImg_path = os.path.join(crop_save_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}{os.path.splitext(image_path)[1]}")
            save(segImg_path, cropImg_path, image_path, None, segImage[230:560,0:700], orig[230:560,0:700])

        # Update the current image index in session state
        st.session_state.current_image_index += 1
        
        # Update the progress bar for user feedback
        progress_bar.progress(st.session_state.current_image_index / total_images)

    # Reset the image index and stop_processing flag for the next run
    st.session_state.current_image_index = 0
    st.session_state.stop_processing = False
    progress_bar.progress(0)

    # Zip the segmented and cropped folders separately for user to download
    seg_zip_name = os.path.join(folder_path, "seg_images.zip")
    crop_zip_name = os.path.join(folder_path, "crop_images.zip")
    shutil.make_archive(seg_zip_name[:-4], 'zip', seg_save_folder)
    shutil.make_archive(crop_zip_name[:-4], 'zip', crop_save_folder)

    # Provide download links for both zipped folders
    seg_zip_url = f"http://192.168.1.65:8502/{os.path.basename(folder_path)}/{os.path.basename(seg_zip_name)}"
    crop_zip_url = f"http://192.168.1.65:8502/{os.path.basename(folder_path)}/{os.path.basename(crop_zip_name)}"

    return seg_zip_url, crop_zip_url

#==========================================================
def save(segImg_path, cropImg_path, path, args, segImage, orig):
    """
    Save the processed images.
    
    Parameters:
    - segImg_path: Path to save the segmented image.
    - cropImg_path: Path to save the cropped image.
    - path: Original image path (not used in this function but kept for compatibility).
    - args: Additional arguments (not used in this function but kept for compatibility).
    - segImage: The segmented image data.
    - orig: The original image data.
    """
    cv2.imwrite(segImg_path, segImage)
    cv2.imwrite(cropImg_path, orig)

#===========================================================
def PavementExtraction(folder_path, model, device):
    """
    Streamlit interface for pavement extraction.
    
    Parameters:
    - folder_path: Path to the folder containing images to be processed.
    - model: The trained model used for image processing.
    - device: The device used for processing (e.g., CPU, GPU).
    """
    if folder_path:
        if os.path.isdir(folder_path):
            # Allow the user to set the threshold value
            threshold_percentage = st.slider("Set the threshold percentage:", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

            # Create directories for saving segmented and cropped images
            seg_save_folder = os.path.join(folder_path, "seg_images")
            crop_save_folder = os.path.join(folder_path, "crop_images")
            os.makedirs(seg_save_folder, exist_ok=True)
            os.makedirs(crop_save_folder, exist_ok=True)

            # Initialize session state variables if they don't exist
            if 'current_image_index' not in st.session_state:
                st.session_state.current_image_index = 0
            if 'stop_processing' not in st.session_state:
                st.session_state.stop_processing = False

            # Start button to begin processing
            if st.button("Start"):
                st.session_state.stop_processing = False
                process_all_images(folder_path, model, seg_save_folder, crop_save_folder, threshold_percentage)

            # Stop button to halt processing
            if st.button("Stop"):
                st.session_state.stop_processing = True

            # After processing all images, allow user to view the processed images
            segmented_images = [f for f in glob.glob(os.path.join(seg_save_folder, '*.[pjP][npP][gG]*'))]
            if segmented_images:
                # Check if 'selected_image' is already in the session state
                if 'selected_image' not in st.session_state:
                    st.session_state.selected_image = segmented_images[0]  # default to the first image

                # Dropdown for user to select an image to view
                st.session_state.selected_image = st.selectbox("Select an image to view:", segmented_images, index=segmented_images.index(st.session_state.selected_image))
    
                # Display the segmented version of the selected image
                seg_image_path = os.path.join(st.session_state.selected_image)
                st.image(seg_image_path, caption="Segmented Image", use_column_width=True)

            # Provide download links for both zipped folders, if they exist
            seg_zip_name = os.path.join(folder_path, "seg_images.zip")
            crop_zip_name = os.path.join(folder_path, "crop_images.zip")
            
            if os.path.exists(seg_zip_name):
                seg_zip_url = f"http://192.168.1.65:8502/{os.path.basename(folder_path)}/{os.path.basename(seg_zip_name)}"
                st.markdown(f"[Download Segmented Images]({seg_zip_url})")
            else:
                st.error("Segmented images zip file does not exist.")

            if os.path.exists(crop_zip_name):
                crop_zip_url = f"http://192.168.1.65:8502/{os.path.basename(folder_path)}/{os.path.basename(crop_zip_name)}"
                st.markdown(f"[Download Cropped Images]({crop_zip_url})")
            else:
                st.error("Cropped images zip file does not exist.")
        else:
            st.error("The folder path is not valid.")
