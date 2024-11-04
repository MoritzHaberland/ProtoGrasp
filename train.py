
from data_loader.ocid_loader import OcidDataset
from modular_configs.configs.dataet_config import get_dataset_config
from torch.utils.data import DataLoader
from model import FasterRCNNModel

def main():
    config = get_dataset_config()

    # Define paths
    split_file_path = config.split_file_path  # Update with your split file path
    root_path = config.root_path  # Update with your images directory

    # Create dataset and dataloader
    dataset = OcidDataset(split_file_path, root_path, "training_0", None)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize model
    num_classes = 32  # Update based on your classes
    model = FasterRCNNModel(num_classes)

    # Train model
    model.train(data_loader, num_epochs=10)

    # Save model
    model.save_model('faster_rcnn_model.pth')

if __name__ == '__main__':
    main()
