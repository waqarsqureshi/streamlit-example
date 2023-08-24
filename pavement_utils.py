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
import time
from datetime import datetime
import glob
# import for image segmentation
from util import get_palette, get_classes
from mmseg.apis import inference_model, init_model

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
    image_with_classes = add_classes_to_image(img, top3_predictions)
    
    return predicted_class_name, image_with_classes, top3_predictions 
#============================================================

#============================================================
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
    else:
        orig = cv2.resize(image,(720, 576)) #(w,h) always remember cv2 takes width and height and numpy use height and width
    return orig
#==========================================================
def classify2(folder_path, model, device):
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
def classify(folder_path, model, device):
    # Check if the folder path is valid
    if folder_path:
        # Get all image paths from the folder
        images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjP][npP][gG]*'), recursive=True)]
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

        # Start button to begin processing
        if st.button("Start"):
            st.session_state.is_running = True

        # Stop button to halt processing
        if st.button("Stop"):
            st.session_state.is_running = False

        # If the session is running and there are images left to process
        if st.session_state.is_running:
            st.write("Processing the images...")

            # Get the current image path and name
            image_path = os.path.join(folder_path, images[st.session_state.image_index])
            
            # Classify the current image
            predicted_class_name, _, top3_predictions = classify_image(image_path, model, device)
            
            # Update session state with the results
            st.session_state.predictions.append(predicted_class_name)
            st.session_state.indices.append(st.session_state.image_index)
            st.session_state.results.append((os.path.basename(image_path), top3_predictions))
            
            # Move to the next image
            st.session_state.image_index += 1
            
            # If there are more images to process, rerun the script
            if st.session_state.image_index < len(images):
                st.experimental_rerun()
            else:
                st.session_state.is_running = False
                st.write("Processing completed!")

        # If all images are processed or processing is stopped, display the graph and results
        if not st.session_state.is_running:
            # Dropdown to select an image
            selected_image_index = st.selectbox("Select an image to view its result:", list(range(len(images))))
            
            if selected_image_index in st.session_state.indices:
                idx = st.session_state.indices.index(selected_image_index)
                image_name, top3_predictions = st.session_state.results[idx]
                _, image_with_classes, _ = classify_image(os.path.join(folder_path, images[idx]), model, device)
                #st.write(os.path.join(folder_path, images[idx]))
                st.image(image_with_classes, caption=image_name)
            else:
                st.write("Image is not processed yet.")
            
            # Plot predictions over time
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
        
            # Save to CSV button
            if st.button("Save Results to CSV"):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                csv_filename = f"{timestamp}_results.csv"
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Image Name", "Prediction 1", "Probability 1", "Prediction 2", "Probability 2", "Prediction 3", "Probability 3"])
                    for result in st.session_state.results:
                        image_name, top3 = result
                        row = [image_name] + [item for sublist in top3 for item in sublist]
                        writer.writerow(row)
                st.success(f"Results saved to {csv_filename}")

    else:
        st.error("The folder path is not valid.")

#============================================================
def gradCam(folder_path, model, device):
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
def process_image(path, model):
    orig = readImage_576x720(path)
    result = inference_model(model, orig)
    result2 = (result.pred_sem_seg.data[0])
    mask = result_segImage(result2, palette=get_palette('roadsurvey'), CLASS_NAMES=get_classes('roadsurvey'))
    segImage = cv2.bitwise_and(orig, mask)
    return segImage, orig
#=============================================================
def should_discard_image(segImage, orig):
    gray_mask = cv2.cvtColor(segImage, cv2.COLOR_BGR2GRAY)
    count = cv2.countNonZero(gray_mask)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thres = (orig.shape[0] * orig.shape[1]) * 0.10
    return count < thres or len(contours) > 25
#=============================================================
def PavementExtraction(folder_path, model, device):
    if folder_path:
        if os.path.isdir(folder_path):
            images = [f for f in glob.glob(os.path.join(folder_path, '**', '*.[pjP][npP][gG]*'), recursive=True)]
            # Sort the image paths
            images.sort()
            image_names = [os.path.basename(image) for image in images]  # Extracting only the image names
            selected_image_name = st.selectbox("Choose an image:", image_names)
            selected_image_index = image_names.index(selected_image_name)
            image_path = images[selected_image_index]  # Get the full path of the selected image
            segImage, orig = process_image(image_path, model)
            
            # Check if the image should be discarded
            if not should_discard_image(segImage, orig):
                # Display the segmented images in Streamlit
                segImage_resized = cv2.resize(segImage, (700, 300))
                st.image(segImage_resized, caption="Pavement Extracted Image", use_column_width=True)
                
                # Save the segmented image
                save_folder = st.text_input("Enter the folder path to save the segmented image:", "")
                if save_folder:
                    if os.path.isdir(save_folder):
                        save_button = st.button("Save Image")
                        if save_button:
                            save_path = os.path.join(save_folder, f"{os.path.splitext(selected_image_name)[0]}_segImg{os.path.splitext(selected_image_name)[1]}")
                            cv2.imwrite(save_path, segImage_resized)
                            st.success(f"Image saved successfully at {save_path}")
                    else:
                        st.error("Please enter a valid folder path.")
            else:
                st.warning("The image was discarded due to poor segmentation results.")




  