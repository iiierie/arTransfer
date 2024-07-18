import torch 
import os
from engine import TransformerNet
from utils import *


def stylize_static_image(content_img_path, model_path, target_width=1080, should_display=False):
    device = torch.device("cpu")  # Load on CPU

    # Load the model
    stylization_model = TransformerNet().to(device)
    training_state = torch.load(model_path, map_location=torch.device('cpu'))  # Load on CPU
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    # Prepare the content image
    img = load_image(content_img_path)
    h, w = img.shape[:2]
    if target_width:
        target_height = int(h * (target_width / w))
        target_shape = (target_height, target_width)
    else:
        target_shape = None
    content_image = prepare_img(content_img_path, target_shape, device)

    # Perform stylization
    with torch.no_grad():
        stylized_img = stylization_model(content_image).cpu().squeeze().permute(1, 2, 0).numpy()

    # Post-process the stylized image
    stylized_img = post_process_image(stylized_img)
    stylized_img = stylized_img.transpose(0,2,1)  # Move channels dimension to the last position
    stylized_img = cv.flip(stylized_img, 1)
    stylized_img = cv.rotate(stylized_img, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Rotate and display if required
    if should_display:
        display_stylized_image(stylized_img)
    
        print(stylized_img.dtype)
        print(stylized_img.shape)

    return stylized_img

# if __name__ == "__main__":
#     stylize_static_image(content_img_path= "cat.jpg", model_path="pretrained_models/mosaic.pth", should_display=True)