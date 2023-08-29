# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire. :heart:

Pavement Surface Condition Index Application
=============================================

Authors: Waqar Shahid Qureshi and Dympna O Sullivan
Affiliation: TU Dublin

Overview
--------
The Pavement Surface Condition Index Application is a robust image processing and classification tool tailored for pavement condition analysis. It leverages deep learning models to classify and segment images, subsequently processing and visualizing the outcomes via the Streamlit framework.

Key Features:
- Pavement Rating
- Pavement Distress Analysis
- Pavement Surface Extraction

Setup & Usage
-------------
1. **Environment Setup**:
   Ensure the necessary libraries and dependencies are installed. Utilize the `requirements.txt` for environment setup:
   ```bash
   pip install -r requirements.txt

Here's the refactored readme.txt for the Pavement Surface Condition Index Application:

markdown

Pavement Surface Condition Index Application
=============================================

Authors: Waqar Shahid Qureshi and Dympna O Sullivan
Affiliation: TU Dublin

Overview
--------
The Pavement Surface Condition Index Application is a robust image processing and classification tool tailored for pavement condition analysis. It leverages deep learning models to classify and segment images, subsequently processing and visualizing the outcomes via the Streamlit framework.

Key Features:
- Pavement Rating
- Pavement Distress Analysis
- Pavement Surface Extraction

Setup & Usage
-------------
1. **Environment Setup**:
   Ensure the necessary libraries and dependencies are installed. Utilize the `requirements.txt` for environment setup:
   ```bash
   pip install -r requirements.txt

    Starting the Application:
        Navigate to the temp directory.
        Initiate an HTTP server:

        python3 -m http.server 8502 --bind 192.168.1.65
    
    Launch the Streamlit application:
    streamlit run streamlit_app.py --server.address 192.168.1.65 --server.port 8501
Dependencies

    streamlit: Web application interface.
    timm, torch, torchvision: Deep learning libraries.
    cv2, PIL: Image processing.
    numpy: Numerical operations.
    datetime, time: Time-related functionalities.
    csv, glob, os, zipfile, shutil: File operations.
    mmseg.apis: Image segmentation.
    pytorch_grad_cam: Gradient-based visual explanations.

Dataset Utilities

Handles datasets, notably Cityscapes and RoadSurvey. Incorporates class names, RGB color palettes for each class, and retrieval functions based on dataset names.
Application Flow

    User selects a pavement analysis process.
    Two logos are displayed.
    "Pavement Surface Condition Index" title appears.
    User uploads a ZIP file containing images.
    ZIP file undergoes:
        Extraction to a distinct directory.
        Image retrieval and processing based on chosen option.

Models

Available on Hugging Face.
Functions & Methods

Detailed descriptions of the main functions and methods driving the application are provided, ranging from dataset utility functions, image processing functions, to segmentation and visualization methods.
Important Notes

    Ensure accurate paths for model checkpoints, configurations, and logos.
    The application leverages session states (st.session_state) to maintain data across reruns, beneficial for preserving model and device details.


This refactored `readme.txt` provides a concise yet comprehensive overview of the application



