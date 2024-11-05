import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights

# Function to get the model
def get_model(num_classes):
    # Load a pre-trained ResNet50 model and get the backbone
    backbone = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))  # Remove the final layers
    backbone.out_channels = 2048  # Set the output channels to match ResNet50

    # Create the anchor generator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    # Create the Faster R-CNN model
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    return model


# Main training function
def main():
    # Define the number of classes (including the background)
    num_classes = 19  # Adjust this according to your dataset

    # Get the model
    model = get_model(num_classes)

    # Move the model to the appropriate devic
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Print the model architecture
    print(model.backbone)

    # Add additional training code here (loading data, training loop, etc.)

if __name__ == '__main__':
    main()