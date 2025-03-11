import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(YOLOv1, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # Darknet-like architecture
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv2
            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv3
            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv4
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Conv5
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        # Detection layers
        self.conv_final = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, grid_size * grid_size * (num_classes + 5 * num_boxes)),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.conv_final(x)
        x = self.fc(x)
        
        # Reshape output to (batch_size, S, S, num_classes + 5*B)
        batch_size = x.size(0)
        x = x.view(batch_size, self.grid_size, self.grid_size, 
                  (self.num_classes + 5 * self.num_boxes))
        return x