import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.yolo import YOLOv1
from data.voc_dataset import VOCDataset,get_train_transforms
from utils.loss import YOLOLoss
import config
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def train():
    # Initialize model
    model = YOLOv1(
        grid_size=config.GRID_SIZE,
        num_boxes=config.NUM_BOXES,
        num_classes=config.NUM_CLASSES,
        pretrained=True
    ).to(config.DEVICE)
    
    # Initialize dataset and dataloader
    train_dataset = VOCDataset(
        root_dir="dataset/VOCdevkit",
        year="2012",
        mode="train",
        transform=get_train_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = VOCDataset(
        root_dir="dataset/VOCdevkit",
        year="2012",
        mode="val"
    )

    val_loader = DataLoader(
        val_dataset,
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
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=config.LEARNING_RATE,
    #     weight_decay=config.WEIGHT_DECAY
    # )

    # Init LR scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
    
    # Create tensorboard summary writer
    writer = SummaryWriter(f"runs/yolov1_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}")

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        lastest_lr = scheduler.get_last_lr()[0]

        optimizer.zero_grad()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            # Forward pass
            predictions = model(images)
            loss, loss_ration = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            if (batch_idx + 1) % config.PARAM_UPDATE_FREQ == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Print statistics
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
                      f"Step [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}",
                      f"LR: {lastest_lr}")
                global_step = epoch*len(train_loader) + batch_idx
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("LR/latest", lastest_lr, global_step)
                writer.add_scalar("LossRatio/pos", loss_ration[0].item(), global_step)
                writer.add_scalar("LossRatio/obj", loss_ration[1].item(), global_step)
                writer.add_scalar("LossRatio/noobj", loss_ration[2].item(), global_step)
                writer.add_scalar("LossRatio/class", loss_ration[3].item(), global_step)
        optimizer.step()

        # Calculate validation loss
        val_loss = 0
        val_ratio = torch.zeros(4)
        if epoch  % 1 == 0:
            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(config.DEVICE)
                    targets = targets.to(config.DEVICE)
                    predictions = model(images)
                    loss, loss_ratio = criterion(predictions, targets)
                    
                    val_loss += loss.item()
                    val_ratio += loss_ratio
            val_loss /= len(val_loader)
            val_ratio /= len(val_loader)
            print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Average Val Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("LossRatio/val_pos", val_ratio[0].item(), epoch)
            writer.add_scalar("LossRatio/val_obj", val_ratio[1].item(), epoch)
            writer.add_scalar("LossRatio/val_noobj", val_ratio[2].item(), epoch)
            writer.add_scalar("LossRatio/val_class", val_ratio[3].item(), epoch)

        scheduler.step()

        # Print epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Average Train Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/avg_train", avg_loss, epoch)
        
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pth")
    
    writer.close()

if __name__ == "__main__":
    train()