import os
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from utils import load_image

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])

class SequentialSubsetSampler(Sampler):
    """Samples elements sequentially, always in the same order from a subset defined by size."""

    def __init__(self, data_source, subset_size):
        assert isinstance(data_source, Dataset) or isinstance(data_source, datasets.ImageFolder)
        self.data_source = data_source

        if subset_size is None:  # if None -> use the whole dataset
            subset_size = len(data_source)
        assert 0 < subset_size <= len(data_source), f'Subset size should be between (0, {len(data_source)}].'
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size

class SimpleDataset(Dataset):
    """Custom dataset for loading and transforming images."""

    def __init__(self, img_dir, target_width):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]

        # Load the first image to determine target height while maintaining aspect ratio
        h, w = load_image(self.img_paths[0]).shape[:2]
        img_height = int(h * (target_width / w))
        self.target_width = target_width
        self.target_height = img_height

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx], target_shape=(self.target_height, self.target_width))
        tensor = self.transform(img)
        return tensor

def get_training_data_loader(training_config, should_normalize=True, is_255_range=False):
    """Creates and returns a DataLoader for the training data."""

    transform_list = [
        transforms.Resize(training_config['image_size']),
        transforms.CenterCrop(training_config['image_size']),
        transforms.ToTensor()
    ]

    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    
    if should_normalize:
        if is_255_range:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL))
        else:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))

    transform = transforms.Compose(transform_list)

    train_dataset = SimpleDataset(training_config['dataset_path'], training_config['image_size'])
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler, drop_last=True)
    
    print(f'Using {len(train_loader)*training_config["batch_size"]} datapoints ({len(train_loader)} batches) (SimpleDataset images) for training.')

    return train_loader
