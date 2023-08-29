# by TU Dublin written by Waqar Shahid Qureshi
#Run the following server in the folder temp folder
#python3 -m http.server 8502 --bind 192.168.1.65
# and run the following in the directory where this code is residing
#streamlit run streamlit_app.py --server.address 192.168.1.65 --server.port 8501


'''
IMPORT necessary libraries

DEFINE paths for model checkpoints and configurations

FUNCTION main():
    - Display Streamlit app title and logos
    - Ask user to select type of analysis (Image Analysis or CSV Data Analysis)
    - IF user selects "Image Analysis":
        - Call image_analysis function
    - ELSE IF user selects "CSV Data Analysis":
        - Call csv_analysis function

FUNCTION image_analysis():
    - Display Streamlit app title
    - Allow user to upload a ZIP file of images
    - IF a ZIP file is uploaded:
        - Create a unique directory based on current timestamp
        - Extract ZIP file to the unique directory
        - Get list of images from the extracted ZIP file
        - IF images are found:
            - IF user selects "Pavement Rating":
                - Initialize model for pavement rating
                - Classify images
            - ELSE IF user selects "Pavement Distress Analysis":
                - Initialize model for pavement distress analysis
                - Analyze pavement distress
            - ELSE IF user selects "Pavement surface Extraction":
                - Initialize model for pavement surface extraction
                - Extract pavement surface
        - ELSE:
            - Display error message in Streamlit
    - ELSE:
        - Display warning message in Streamlit

FUNCTION csv_analysis():
    - Display Streamlit app title
    - Define user-defined parameters using Streamlit sidebar
    - Allow user to upload a CSV file
    - IF a CSV file is uploaded:
        - Create a unique directory based on current timestamp
        - Save uploaded CSV file to the unique directory
        - Read CSV file from the unique directory
        - IF CSV file has required columns:
            - Adjust predictions based on probability threshold
            - Apply chosen filter to data
            - Determine most common rating for each section size
            - Determine x-axis values based on user's choice
            - Create interactive plot using Plotly
            - Display plot in Streamlit
            - IF user clicks "Save Data to CSV" button:
                - Compute specified information for CSV
                - Save data to CSV file
                - Generate a URL link for downloading the CSV file
        - ELSE:
            - Display error message in Streamlit

IF __name__ == "__main__":
    - Call main function

'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from datetime import datetime
import glob, os
from csv_analysis import csv_analysis
from image_analysis import image_analysis

# Define paths for model checkpoints and configurations
checkpoint_path_swin = '/home/pms/pms/pms-code/ipsci-script/checkpoints-30052023/10-class/20230428-105301-swinv2_base_window12to24_192to384_22kft1k-384/last.pth.tar'
config_deeplabv3plus = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/deeplabv3plus_r50b-d8_4xb2-160k_roadsurvey-512x512.py"
check_point_deeplabv3plus = "/home/pms/pms/pms-code/ipsci-script/checkpoints-1/deeplabv3plus_r50-d8_512x512_160k_new/iter_160000.pth"

def display_help():
    st.subheader("User Help Guide for Pavement Surface Condition Index Application")
    st.write("""
    ### Welcome to the Pavement Surface Condition Index Application!

    
1. The Application:
This application, developed by TU Dublin and authored by Waqar Shahid Qureshi, provides a comprehensive solution for analyzing pavement conditions. Here's a step-by-step guide to help you navigate and use the application effectively:

2. Main Interface:

Upon launching, you'll be greeted with the title "Pavement Surface Condition Index" and logos of TU Dublin and PMS.

    Analysis Selection: Use the dropdown menu labeled "Select the type of Analysis" to choose between "Image Analysis" and "CSV Data Analysis".

3. Image Analysis:

If you select "Image Analysis":

    Upload ZIP File: Use the provided interface to upload a ZIP file containing the images you wish to analyze.

    Analysis Type: Once the ZIP file is uploaded, you'll be prompted to select the specific type of image analysis:
        Pavement Rating: Classifies the images based on pavement rating.
        Pavement Distress Analysis: Analyzes the distress in the pavement.
        Pavement Surface Extraction: Extracts the pavement surface from the images.

4. CSV Data Analysis:

If you select "CSV Data Analysis":

    Parameter Definition: On the sidebar, you can define various parameters that will influence the analysis.

    Upload CSV File: Use the provided interface to upload a CSV file containing the data you wish to analyze.

    Interactive Plot: Once the CSV file is uploaded and processed, an interactive plot will be displayed based on the data. You can hover over the plot for detailed insights.

    Save Data: If you wish to save the processed data, click on the "Save Data to CSV" button. A link will be generated, allowing you to download the CSV file.

5. Troubleshooting:

    Invalid ZIP File: If no images are found in the uploaded ZIP file, an error message will be displayed. Ensure the ZIP file contains valid images and try again.

    Invalid CSV File: If the uploaded CSV file doesn't have the required columns, an error message will be displayed. Ensure the CSV file is in the correct format and try again.

6. Feedback & Support:

If you encounter any issues or have suggestions for improvements, please reach out to the development team. Your feedback is invaluable to us!

    Thank you for using the Pavement Surface Condition Index Application. We hope it serves your needs effectively and provides valuable insights into pavement conditions!
    """)

def main():
    st.title("iPSCI Application")
    # Display logos in two columns
    left_column, right_column = st.columns(2)
    left_column.image("/home/pms/Pictures/TU Dublin Logo_resized.jpg", width=200)
    right_column.image("/home/pms/Pictures/pms.png", width=200)
        # Display the title of the application
    
    # Checkboxes for user confirmation on paths
    st.subheader("Configuration Confirmation")
    checkpoint_confirmed = st.checkbox("I have set the models checkpoint paths for both the models or I'm okay with using the default paths.")
    config_confirmed = st.checkbox("I have set the pavement extraction model configuration path or I'm okay with using the default paths.")
    config_confirmed = st.checkbox("I will read the Help first if I am the first time user of the application.")
    config_confirmed = st.checkbox("Make sure the file server is running on the server machine: python3 -m http.server 8502 --bind 192.168.1.65")
    
    # Only show the dropdown if both checkboxes are ticked
    if checkpoint_confirmed and config_confirmed:
        # Ask the user for the type of analysis or to view the help
        analysis_type = st.selectbox("Select an option:", [" ","Help", "Image Analysis", "CSV Data Analysis"])

        if analysis_type == "Image Analysis":
            image_analysis()
        elif analysis_type == "CSV Data Analysis":
            csv_analysis()
        elif analysis_type == "Help":
            display_help()
    else:
        st.warning("Please confirm the paths for checkpoints and configurations to proceed.")

if __name__ == "__main__":
    main()
