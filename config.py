import torch

# Dataset Parameters
PASCAL_VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(PASCAL_VOC_CLASSES)

# Model Parameters
IMAGE_SIZE = 448  # Input image size for YOLO
GRID_SIZE = 7     # Grid size (SxS)
NUM_BOXES = 2     # Number of bounding boxes per grid cell
NUM_CHANNELS = 3  # Input channels

# Training Parameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
NUM_EPOCHS = 100
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss Parameters
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5