import torch
from torchvision import models
from collections import namedtuple

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, show_progress=False):
        super(Vgg16, self).__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg16 = models.vgg16(weights=weights, progress=show_progress).eval()
        vgg_pretrained_features = vgg16.features
        
        self.slice1 = torch.nn.Sequential(*vgg_pretrained_features[:4])
        self.slice2 = torch.nn.Sequential(*vgg_pretrained_features[4:9])
        self.slice3 = torch.nn.Sequential(*vgg_pretrained_features[9:16])
        self.slice4 = torch.nn.Sequential(*vgg_pretrained_features[16:23])
        
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        relu1_2 = self.slice1(x)
        relu2_2 = self.slice2(relu1_2)
        relu3_3 = self.slice3(relu2_2)
        relu4_3 = self.slice4(relu3_3)
        
        vgg_outputs = namedtuple("VggOutputs", self.layer_names)
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out

# Set the perceptual loss network to be VGG16
PerceptualLossNet = Vgg16

# # Function to initialize and load the PerceptualLossNet model
# def load_perceptual_loss_net():
#     model = PerceptualLossNet()
#     return model

# # Example of using the model
# if __name__ == '__main__':
#     model = load_perceptual_loss_net()
#     input_image = torch.randn(1, 3, 256, 256)  # Example input
#     features = model(input_image)
#     for name, feature in zip(model.layer_names, features):
#         print(f"{name}: {feature.shape}")
