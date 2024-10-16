from ml_collections import ConfigDict

def get_training_config():
    training = ConfigDict()
    training.learning_rate = 0.001
    training.weight_decay = 0.0001
    training.batch_size = 16
    training.num_epochs = 50
    return training