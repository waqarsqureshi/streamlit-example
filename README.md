Pavement Surface Condition Index Application
Table of Contents

    Author and Affiliation
    Overview
    Pre-requisites
    Dataset Utilities
    Functions Breakdown
    Important Notes

Author and Affiliation

    Author: Waqar Shahid Qureshi
    Affiliation: TU Dublin

Overview

This application is designed to analyze pavement conditions using images. It offers three main functionalities:

    Pavement Rating
    Pavement Distress Analysis
    Pavement Surface Extraction

Pre-requisites
Setting up the Environment

Before running the application, ensure that the required libraries and dependencies are installed. Use the requirements.txt file to set up the environment:

bash

pip install -r requirements.txt

Ensure you are in the temp folder and start an HTTP server using:

bash

python3 -m http.server 8502 --bind 192.168.1.65

Run the Streamlit application using:

bash

streamlit run streamlit_app.py --server.address 192.168.1.65 --server.port 8501

Dataset Utilities

Copyright (c) OpenMMLab. All rights reserved.

This section provides utility functions for handling different datasets, specifically Cityscapes and RoadSurvey. It includes class names for each dataset, color palettes (RGB) for each class in the datasets, and functions to retrieve class names and palettes based on the dataset name.

Functions and Methods:

    main():
        Purpose: The main driver of the application. It manages the user interface, ZIP file extraction, and invokes the appropriate pavement analysis functions based on user choice.
        Input Arguments: None.
        Output: None (directly interacts with the Streamlit interface).

    cityscapes_classes():
        Purpose: Provides class names for the Cityscapes dataset.
        Input Arguments: None.
        Output: List of class names for the Cityscapes dataset.

    roadsurvey_classes():
        Purpose: Provides class names for the RoadSurvey dataset.
        Input Arguments: None.
        Output: List of class names for the RoadSurvey dataset.

    cityscapes_palette():
        Purpose: Provides the RGB color palette for the Cityscapes dataset.
        Input Arguments: None.
        Output: RGB color palette for the Cityscapes dataset.

    roadsurvey_palette():
        Purpose: Provides the RGB color palette for the RoadSurvey dataset.
        Input Arguments: None.
        Output: RGB color palette for the RoadSurvey dataset.

    get_classes(dataset):
        Purpose: Retrieves class names for a given dataset.
        Input Arguments:
            dataset (str): Name of the dataset.
        Output: List of class names for the specified dataset.

    get_palette(dataset):
        Purpose: Retrieves the RGB color palette for a given dataset.
        Input Arguments:
            dataset (str): Name of the dataset.
        Output: RGB color palette for the specified dataset.

    get_transform(size, NORMALIZE_MEAN, NORMALIZE_STD):
        Purpose: Returns the image transformation pipeline.
        Input Arguments:
            size (int): Desired image size.
            NORMALIZE_MEAN (list): Normalization mean values.
            NORMALIZE_STD (list): Normalization standard deviation values.
        Output: A composed transformation for image preprocessing.

    reshape_transform(tensor, height, width):
        Purpose: Reshapes the tensor for visualization.
        Input Arguments:
            tensor (Tensor): Input tensor.
            height (int): Desired height.
            width (int): Desired width.
        Output: Reshaped tensor.

    create_load_model(name, checkpoint_path):
        Purpose: Creates and loads a model with a given checkpoint.
        Input Arguments:
            name (str): Model name.
            checkpoint_path (str): Path to the model checkpoint.
        Output: Loaded model.

    add_classes_to_image(image, top3_predictions):
        Purpose: Adds class predictions to an image.
        Input Arguments:
            image (Image): Input image.
            top3_predictions (list): Top 3 class predictions.
        Output: Image with class annotations.

    classify_image(image_path, model, device):
        Purpose: Classifies a single image.
        Input Arguments:
            image_path (str): Path to the image.
            model (Model): Trained model.
            device (Device): Computation device (CPU/GPU).
        Output: Predicted class name and top 3 predictions.

    classify2(folder_path, model, device):
        Purpose: Processes images one by one and displays results.
        Input Arguments:
            folder_path (str): Path to the folder containing images.
            model (Model): Trained model.
            device (Device): Computation device.
        Output: None (directly interacts with the Streamlit interface).

    process_single_image(image_path, model, device):
        Purpose: Processes a single image.
        Input Arguments:
            image_path (str): Path to the image.
            model (Model): Trained model.
            device (Device): Computation device.
        Output: Predicted class name and top 3 predictions.

    update_progress_bar(progress_bar, current_index, total_images):
        Purpose: Updates the progress bar in the Streamlit app.
        Input Arguments:
            progress_bar (Progress Bar): Streamlit progress bar.
            current_index (int): Current image index.
            total_images (int): Total number of images.
        Output: None (updates the Streamlit progress bar).

    display_results(images, folder_path):
        Purpose: Displays the results of the processed images.
        Input Arguments:
            images (list): List of processed images.
            folder_path (str): Path to the folder containing images.
        Output: None (directly interacts with the Streamlit interface).

    plot_predictions():
        Purpose: Plots the predictions over time.
        Input Arguments: None.
        Output: None (directly interacts with the Streamlit interface).

    save_to_csv(folder_path):
        Purpose: Saves the results to a CSV file.
        Input Arguments:
            folder_path (str): Path to the folder containing images.
        Output: None (saves results to a CSV file).

    classify(folder_path, model, device):
        Purpose: Classifies all images in a folder.
        Input Arguments:
            folder_path (str): Path to the folder containing images.
            model (Model): Trained model.
            device (Device): Computation device.
        Output: None (directly interacts with the Streamlit interface).

    gradCam(folder_path, model, device):
        Purpose: Applies Grad-CAM visualization to images.
        Input Arguments:
            folder_path (str): Path to the folder containing images.
            model (Model): Trained model.
            device (Device): Computation device.
        Output: None (directly interacts with the Streamlit interface).

    initialize_model(model_name, checkpointPath, config):
        Purpose: Initializes and loads a model.
        Input Arguments:
            model_name (str): Model name.
            checkpointPath (str): Path to the model checkpoint.
            config (str, optional): Model configuration.
        Output: Initialized model and computation device.

    readImage_576x720(path):
        Purpose: Reads and resizes an image.
        Input Arguments:
            path (str): Path to the image.
        Output: Resized image.

    result_segImage(result, palette, CLASS_NAMES):
        Purpose: Computes the segmentation mask for an image.
        Input Arguments:
            result (Result): Segmentation result.
            palette (list): Color palette.
            CLASS_NAMES (list): Class names.
        Output: Segmentation mask.

    process_image(image_path, model):
        Purpose: Processes an image for segmentation.
        Input Arguments:
            image_path (str): Path to the image.
            model (Model): Trained model.
        Output: Segmented image and original image.

    should_discard_image(segImage, orig, threshold_percentage):
        Purpose: Determines if an image should be discarded based on segmentation results.
        Input Arguments:
            segImage (Image): Segmented image.
            orig (Image): Original image.
            threshold_percentage (float): Threshold percentage for discarding.
        Output: Boolean indicating if the image should be discarded.

    PavementExtraction2(folder_path, model, device):
        Purpose: Processes images for pavement extraction.
        Input Arguments:
            folder_path (str): Path to the folder containing images.
            model (Model): Trained model.
            device (Device): Computation device.
        Output: None (directly interacts with the Streamlit interface).

    zip_folder(folder_path, zip_name):
        Purpose: Zips a folder.
        Input Arguments:
            folder_path (str): Path to the folder.
            zip_name (str): Desired name for the zip file.
        Output: None (creates a zip file).


Important Notes

    Ensure that the paths for model checkpoints, configurations, and logos are correctly set.
    The application uses session states (st.session_state) to persist data across reruns. This is especially useful for storing the model and device information once initialized, so they don't have to be reloaded on every interaction.
