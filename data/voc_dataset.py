import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import PASCAL_VOC_CLASSES, IMAGE_SIZE, GRID_SIZE

class VOCDataset(Dataset):
    def __init__(self, root_dir, year="2012", mode="train", transform=None):
        self.root_dir = root_dir
        self.year = year
        self.mode = mode
        self.transform = transform if transform else self._get_default_transform()
        
        # Get image and annotation paths
        self.images_dir = os.path.join(root_dir, f"VOC{year}", "JPEGImages")
        self.annotations_dir = os.path.join(root_dir, f"VOC{year}", "Annotations")
        
        # Get list of image IDs
        split_file = os.path.join(
            root_dir, f"VOC{year}", "ImageSets", "Main", f"{mode}.txt"
        )
        with open(split_file, "r") as f:
            self.image_ids = [x.strip() for x in f.readlines()]
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(PASCAL_VOC_CLASSES)}
        
    def add_padding(self, img):
        """Add padding to make image square while maintaining aspect ratio"""
        w, h = img.size
        dim_diff = abs(h - w)
        
        # Find padding dimensions
        pad1, pad2 = dim_diff // 2, dim_diff - (dim_diff // 2)
        
        # Add padding
        if h <= w:
            padding = (0, pad1, 0, pad2)  # left, top, right, bottom
        else:
            padding = (pad1, 0, pad2, 0)
        
        # Add padding and resize
        img = transforms.Pad(padding, fill=0)(img)  # Add padding with black
        img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)
        
        return img, padding, (w, h)

    def adjust_boxes_for_padding(self, boxes, orig_size, padding):
        """
        Adjust box coordinates for padding
        boxes: tensor of shape (N, 4) with values normalized to [0, 1]
            format is (x_center, y_center, width, height)
        orig_size: original image size (width, height)
        padding: tuple of (pad_left, pad_top, pad_right, pad_bottom)
        """
        w, h = orig_size
        pad_left, pad_top, pad_right, pad_bottom = padding
        
        # Calculate effective dimensions after padding
        effective_w = w + pad_left + pad_right
        effective_h = h + pad_top + pad_bottom
        
        # Adjust coordinates
        adjusted_boxes = boxes.clone()
        if len(boxes) > 0:
            # Convert x, y from relative [0, 1] to absolute coordinates
            x_center = boxes[:, 0] * w
            y_center = boxes[:, 1] * h
            
            # Add padding offsets to center coordinates only
            x_center += pad_left
            y_center += pad_top
            
            # Convert back to relative coordinates in padded image space
            adjusted_boxes[:, 0] = x_center / effective_w  # x center
            adjusted_boxes[:, 1] = y_center / effective_h  # y center
            
            # Width and height need to be scaled relative to the new image size
            # but maintaining their absolute size
            adjusted_boxes[:, 2] = (boxes[:, 2] * w) / effective_w  # width
            adjusted_boxes[:, 3] = (boxes[:, 3] * h) / effective_h  # height
        
        return adjusted_boxes
    
    def _get_default_transform(self):
        """Only normalize the image, resizing is handled in add_padding"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _get_annotation(self, annotation_path, orig_image_size):
        """Get annotation boxes in relative coordinates"""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        image_width, image_height = orig_image_size
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.class_to_idx:
                continue
                
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / image_width
            ymin = float(bbox.find("ymin").text) / image_height
            xmax = float(bbox.find("xmax").text) / image_width
            ymax = float(bbox.find("ymax").text) / image_height
            
            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            
            boxes.append([x_center, y_center, width, height])
            labels.append(self.class_to_idx[label])
            
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels)
    
    def _convert_to_yolo_format(self, boxes, labels):
        """Convert boxes to YOLO target format"""
        target = torch.zeros((GRID_SIZE, GRID_SIZE, 25))  # 20 classes + 5 box values
        
        if len(boxes) == 0:
            return target
            
        for box_idx in range(len(boxes)):
            box = boxes[box_idx]
            label = labels[box_idx]
            
            # Get grid cell indices
            grid_x = int(box[0] * GRID_SIZE)
            grid_y = int(box[1] * GRID_SIZE)
            
            # Convert box coordinates relative to grid cell
            x = box[0] * GRID_SIZE - grid_x
            y = box[1] * GRID_SIZE - grid_y
            w = box[2]
            h = box[3]
            
            if grid_x < GRID_SIZE and grid_y < GRID_SIZE:
                if target[grid_y, grid_x, 4] == 0:  # No object in this cell yet
                    target[grid_y, grid_x, 0:4] = torch.tensor([x, y, w, h])
                    target[grid_y, grid_x, 4] = 1  # objectness score
                    target[grid_y, grid_x, 5 + label] = 1  # class label one-hot
                    
        return target
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        img = Image.open(image_path).convert("RGB")
        orig_size = img.size  # Save original size
        
        # Add padding and get transform info
        img, padding, orig_size = self.add_padding(img)
        
        # Load annotations
        annotation_path = os.path.join(self.annotations_dir, f"{image_id}.xml")
        boxes, labels = self._get_annotation(annotation_path, orig_size)
        
        # Adjust boxes for padding
        boxes = self.adjust_boxes_for_padding(boxes, orig_size, padding)
        
        # Apply remaining transforms (normalization)
        if self.transform:
            img = self.transform(img)
        
        # Convert to YOLO format
        target = self._convert_to_yolo_format(boxes, labels)
        
        return img, target

# Data augmentation transforms for training
def get_train_transforms():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Simple transforms for validation/testing
def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])