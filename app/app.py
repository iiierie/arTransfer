import streamlit as st
import os
from PIL import Image
import tempfile
import shutil
import sys
import io
import base64
import numpy as np
import cv2



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylize import stylize_static_image 

def get_download_link(img_array, filename):
    buffer = io.BytesIO()
    img_pil = Image.fromarray(img_array)
    img_pil.save(buffer, format='JPEG')
    b64 = base64.b64encode(buffer.getvalue()).decode()

    # Style the download button
    download_link = f'<a href="data:file/jpg;base64,{b64}" download="{filename}.jpg" style="text-decoration: none; color: white; background-color: #008CBA; padding: 10px 20px; border-radius: 5px; display: inline-block;">Download Stylized Image</a>'
    
    return download_link


def resize_image(image_path, target_size=(1080, 1080)):
    img = Image.open(image_path)
    img_resized = img.resize(target_size)
    return img_resized

def main():
    st.title('arTransfer - a Style Transfer App')

    
    st.sidebar.title('Options')
    content_image = st.sidebar.file_uploader('Upload Content Image', type=['jpg', 'jpeg', 'png'])

    st.sidebar.markdown('### Choose Style Preset')

    style_images = {
        "mosaic": "../presets/mosaic.jpg",
        "candy": "../presets/candy.jpg",
        "starry_night": "../presets/starry_night.jpg",
        "edtaonisl": "../presets/edtaonisl.jpg"
    }

    
    style_choice = st.sidebar.selectbox('Choose a style', list(style_images.keys()))

    # Resize style images to a uniform size
    resized_images = {style: resize_image(image_path) for style, image_path in style_images.items()}

    # Display style presets in a 2x2 grid with equal-sized images
    cols = st.sidebar.columns(2)
    for i, (style, img_resized) in enumerate(resized_images.items()):
        with cols[i % 2]:  # Alternate between two columns
            st.image(img_resized, caption=style, use_column_width=True)



    if st.sidebar.button('Stylize Image') and content_image is not None:
        # Display a loading message while stylizing
        with st.spinner('Stylizing image...'):
            # Save the uploaded image temporarily
            temp_dir = tempfile.mkdtemp()
            content_img_path = os.path.join(temp_dir, content_image.name)
            
            with open(content_img_path, 'wb') as f:
                f.write(content_image.getbuffer())

            model_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained_models', f'{style_choice}.pth')

            # Perform stylization
            stylized_img = stylize_static_image(content_img_path, model_path)

            # Display stylized image
            st.subheader('Stylized Image')
            stylized_img_array = np.array(stylized_img)
            stylized_img_array = stylized_img_array[:, :, ::-1]  # Convert RGB to BGR for correct display
            st.image(stylized_img_array, caption='Stylized Image', use_column_width=True)

            # Display comparison section
            st.subheader('Comparison')
            col1, col2 = st.columns(2)
            col1.header("Original Image")
            col1.image(content_image, use_column_width=True)

            col2.header(f"({style_choice})")
            col2.image(stylized_img_array, caption='Stylized Image', use_column_width=True)

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            
            st.markdown(get_download_link(stylized_img_array, 'stylized_image'), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
