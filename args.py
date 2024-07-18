import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_images_path", type=str, help="Path to directory containing style images", default = '../style_images')
    parser.add_argument("--num_of_epochs", type=int, help="Number of training epochs", default=10)
    parser.add_argument("--dataset_path", type=str, help="Path to MS COCO dataset", default='../datasets/mscoco')
    parser.add_argument("--image_size", type=int, help="Image size for training", default=256)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=8)
    parser.add_argument("--subset_size", type=int, help="Number of images from MS COCO to use", default=None)
    parser.add_argument("--model_binaries_path", type=str, help="Path to save model binaries", default='../models/binaries')
    parser.add_argument("--checkpoints_path", type=str, help="Path to save checkpoints", default='../models/checkpoints')
    return parser.parse_args()