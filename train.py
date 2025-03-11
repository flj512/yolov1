import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.yolo import YOLOv1
from data.voc_dataset import VOCDataset
from utils.loss import YOLOLoss
import config

def train():
    # Initialize model
    model = YOLOv1(
        grid_size=config.GRID_SIZE,
        num_boxes=config.NUM_BOXES,
        num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    
    # Initialize dataset and dataloader
    train_dataset = VOCDataset(
        root_dir="dataset/VOCdevkit",
        year="2012",
        mode="train"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize loss and optimizer
    criterion = YOLOLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print statistics
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{config.NUM_EPOCHS}], "
                      f"Step [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Print epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()