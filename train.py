import os
import torch
from torch.optim import Adam
import numpy as np
from datasets import *
import utils 
from perpetual_loss import PerceptualLossNet
from engine import TransformerNet
from args import parse_args

def train(training_config, style_img_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = os.path.join(training_config.checkpoints_path, f'{style_img_name}_training_log.txt')

    train_loader = utils.get_training_data_loader(training_config)
    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)
    optimizer = Adam(transformer_net.parameters())

    style_img_path = os.path.join(training_config.style_images_path, style_img_name)
    style_img = utils.prepare_img(style_img_path, target_shape=None, device=device, batch_size=training_config.batch_size)
    style_img_features = perceptual_loss_net(style_img)
    target_style_repr = [utils.gram_matrix(x) for x in style_img_features]

    utils.print_header(training_config)

    for epoch in range(training_config.num_of_epochs):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            content_batch = content_batch.to(device)
            stylized_batch = transformer_net(content_batch)

            content_features = perceptual_loss_net(content_batch)
            stylized_features = perceptual_loss_net(stylized_batch)

            content_loss = torch.nn.MSELoss(reduction='mean')(content_features.relu2_2, stylized_features.relu2_2)
            stylized_style_repr = [utils.gram_matrix(x) for x in stylized_features]
            style_loss = sum(torch.nn.MSELoss(reduction='mean')(gt, hat) for gt, hat in zip(target_style_repr, stylized_style_repr)) / len(target_style_repr) * training_config.style_weight
            tv_loss = training_config.tv_weight * utils.total_variation(stylized_batch)

            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_message = (f'Epoch [{epoch + 1}/{training_config.num_of_epochs}]\t'
                           f'Batch [{batch_id + 1}/{len(train_loader)}]\t'
                           f'Content Loss: {content_loss.item():.4f}\t'
                           f'Style Loss: {style_loss.item():.4f}\t'
                           f'TV Loss: {tv_loss.item():.4f}\t'
                           f'Total Loss: {total_loss.item():.4f}')

            # Print to console
            print(log_message)
            # Write to log file
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')

            if training_config.checkpoint_freq is not None and (batch_id + 1) % training_config.checkpoint_freq == 0:
                save_checkpoint(training_config, transformer_net, optimizer, epoch, batch_id, style_img_name)

    save_final_model(training_config, transformer_net, optimizer, style_img_name)

def save_checkpoint(training_config, model, optimizer, epoch, batch_id, style_img_name):
    checkpoint = {
        'epoch': epoch,
        'batch_id': batch_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_name = f"checkpoint_{style_img_name.split('.')[0]}_epoch_{epoch}_batch_{batch_id}.pth"
    checkpoint_path = os.path.join(training_config.checkpoints_path, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)

def save_final_model(training_config, model, optimizer, style_img_name):
    model_metadata = utils.get_training_metadata(training_config)
    model_metadata['state_dict'] = model.state_dict()
    model_metadata['optimizer_state_dict'] = optimizer.state_dict()
    model_name = f"final_model_{style_img_name.split('.')[0]}.pth"
    model_path = os.path.join(training_config.model_binaries_path, model_name)
    torch.save(model_metadata, model_path)

if __name__ == "__main__":
    args = parse_args()
    args.dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
    args.style_images_path = os.path.join(os.path.dirname(__file__), 'data', 'style-images')
    args.model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    args.checkpoints_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    args.image_size = 256  # Example image size
    args.batch_size = 8  # Example batch size
    args.subset_size = None  # Use the whole dataset

    os.makedirs(args.model_binaries_path, exist_ok=True)
    os.makedirs(args.checkpoints_path, exist_ok=True)

    style_images = [f for f in os.listdir(args.style_images_path) if os.path.isfile(os.path.join(args.style_images_path, f))]

    for style_img_name in style_images:
        print(f"Training with style image: {style_img_name}")
        os.makedirs(os.path.join(args.checkpoints_path, style_img_name.split('.')[0]), exist_ok=True)
        train(args, style_img_name)

