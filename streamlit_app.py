# Import necessary libraries
import streamlit as st
import zipfile
from datetime import datetime
import glob, os
from pavement_utils import (classify, gradCam, PavementExtraction, initialize_model)

# Define paths for model checkpoints and configurations
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'
config_deeplabv3plus = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512.py"
check_point_deeplabv3plus = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/iter_160000.pth"

def main():
    # Define the options for pavement analysis
    options = ["Pavement Rating", "Pavement Distress Analysis", "Pavement surface Extraction"]
    selected_option = st.sidebar.selectbox("Choose a process:", options)
    
    # Display logos in two columns
    left_column, right_column = st.columns(2)
    left_column.image("/home/pms/Pictures/TU Dublin Logo_resized.jpg", width=200)
    right_column.image("/home/pms/Pictures/pms.png", width=200)
    
    # Display the title of the application
    st.title("Pavement Surface Condition Index")
    
    # Allow users to upload a ZIP file
    uploaded_zip = st.file_uploader("Upload a ZIP file of images:", type=['zip'])

    # If a ZIP file is uploaded
    if uploaded_zip:
        temp_dir_path = "/home/pms/streamlit-example/temp/"
        os.makedirs(temp_dir_path, exist_ok=True)

        # Check if the unique_temp_dir is already in the session state
        if 'unique_temp_dir' not in st.session_state:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # Get the current date and time
            unique_temp_dir = os.path.join(temp_dir_path, timestamp)  # Use the timestamp as the directory name
            os.makedirs(unique_temp_dir)
            
            # Extract the ZIP file
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(unique_temp_dir)
                st.session_state.unique_temp_dir = unique_temp_dir  # Store the directory path in the session state
            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP file. Please upload a valid ZIP file.")

        # Get the list of images from the extracted ZIP file
        images = [f for f in glob.glob(os.path.join(st.session_state.unique_temp_dir, '**', '*.[pjP][npP][gG]*'), recursive=True)]
        
        # If images are found
        if images:
            # For each option, initialize the model and process the images accordingly
            if selected_option == "Pavement Rating":
                modelName = 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
                st.session_state.model, st.session_state.device = initialize_model(modelName, checkpoint_path_swin)
                classify(st.session_state.unique_temp_dir, st.session_state.model, st.session_state.device)

            elif selected_option == "Pavement Distress Analysis":
                modelName = 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k'
                st.session_state.model, st.session_state.device = initialize_model(modelName, checkpoint_path_swin)
                gradCam(st.session_state.unique_temp_dir, st.session_state.model, st.session_state.device)

            elif selected_option == "Pavement surface Extraction":
                modelName = 'deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512'
                st.session_state.model, st.session_state.device = initialize_model(modelName, check_point_deeplabv3plus, config_deeplabv3plus)
                PavementExtraction(st.session_state.unique_temp_dir, st.session_state.model, st.session_state.device)
        else:
            st.error("No images found in the ZIP file. Please upload a valid ZIP file containing images.")
    else:
        st.warning("Please upload a ZIP file containing images.")

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
