import matplotlib.pyplot as plt
import torch 
import os
import os
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2 as cv

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_255 = IMAGENET_MEAN_1 * 255
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # Convert BGR (opencv format) into RGB

    if target_shape is not None:  # Resize section
        if isinstance(target_shape, int) and target_shape != -1:  # Scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # Set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32)  # Convert from uint8 to float32
    img /= 255.0  # Get to [0, 1] range
    return img

def post_process_image(dump_img):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy image got {type(dump_img)}'
    mean = IMAGENET_MEAN_1.reshape(1, 1, -1)  # Add new axis
    std = IMAGENET_STD_1.reshape(1, 1, -1)  # Add new axis
    dump_img = (dump_img * std) + mean  # De-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)  # Move color channels to the last dimension
    return dump_img


def save_and_maybe_display_image(dump_img, img_name, output_dir, should_display):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'
    dump_img = post_process_image(dump_img)
    # print(dump_img.shape)
    dump_img = dump_img.transpose(0,2,1)  # Move channels dimension to the last position
    dump_img = cv.rotate(dump_img, cv.ROTATE_90_CLOCKWISE)
    # print(dump_img.shape)
    assert dump_img.shape[2] == 3, f'Expected 3 channels, got {dump_img.shape[2]}'
    assert dump_img.dtype == np.uint8, f'Expected uint8 dtype, got {dump_img.dtype}'
    os.makedirs(output_dir, exist_ok=True)
    cv.imwrite(os.path.join(output_dir, img_name), dump_img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...
    if should_display:
        plt.imshow(dump_img)
        plt.show()


def prepare_img(img_path, target_shape, device, should_normalize=True, is_255_range=False):
    img = load_image(img_path, target_shape=target_shape)
    transform_list = [transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return img

def display_stylized_image(stylized_img):
    # Display the image in a popup window using Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(stylized_img[:, :, ::-1])  # Convert BGR to RGB for display
    plt.axis('off')  # Turn off axis
    plt.show()  # Show the image in a popup window


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram

# print(gram_matrix(torch.randn(4, 3, 256, 256)).shape)