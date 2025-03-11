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
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        
        # Load annotations
        annotation_path = os.path.join(self.annotations_dir, f"{image_id}.xml")
        boxes, labels = self._get_annotation(annotation_path)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        # Convert to YOLO format
        target = self._convert_to_yolo_format(boxes, labels)
        
        return image, target
    
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _get_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find("size")
        image_width = float(size.find("width").text)
        image_height = float(size.find("height").text)
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.class_to_idx:
                continue
                
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            
            # Normalize coordinates
            xmin, xmax = xmin/image_width, xmax/image_width
            ymin, ymax = ymin/image_height, ymax/image_height
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[label])
            
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels)
    
    def _convert_to_yolo_format(self, boxes, labels):
        target = torch.zeros((GRID_SIZE, GRID_SIZE, 25))  # 20 classes + 5 box values
        
        if len(boxes) == 0:
            return target
            
        # Convert from (xmin, ymin, xmax, ymax) to (x, y, w, h)
        box_centers = (boxes[:, 2:] + boxes[:, :2]) / 2
        box_sizes = boxes[:, 2:] - boxes[:, :2]
        
        for box_idx in range(len(boxes)):
            box_center = box_centers[box_idx]
            box_size = box_sizes[box_idx]
            label = labels[box_idx]
            
            # Get grid cell indices
            grid_x = int(box_center[0] * GRID_SIZE)
            grid_y = int(box_center[1] * GRID_SIZE)
            
            # Convert box coordinates relative to grid cell
            x = box_center[0] * GRID_SIZE - grid_x
            y = box_center[1] * GRID_SIZE - grid_y
            w = box_size[0]
            h = box_size[1]
            
            if grid_x < GRID_SIZE and grid_y < GRID_SIZE:
                if target[grid_y, grid_x, 4] == 0:  # No object in this cell yet
                    target[grid_y, grid_x, 0:4] = torch.tensor([x, y, w, h])
                    target[grid_y, grid_x, 4] = 1  # objectness score
                    target[grid_y, grid_x, 5 + label] = 1  # class label one-hot
                    
        return target