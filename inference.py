import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.yolo import YOLOv1
from utils.utils import convert_cellboxes, non_max_suppression
from config import *
import time
from datetime import datetime
from utils.utils import add_padding

class YOLOPredictor:
    def __init__(self, checkpoint_path, conf_threshold=0.5, nms_threshold=0.4):
        self.device = DEVICE
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize model
        self.model = YOLOv1(
            grid_size=GRID_SIZE,
            num_boxes=NUM_BOXES,
            num_classes=NUM_CLASSES,
            pretrained=True
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform for input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, img):
        """Convert OpenCV image to PIL Image and apply transforms"""
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        img, padding, _ = add_padding(img)
        return self.transform(img).unsqueeze(0), padding
    
    def process_predictions(self, predictions, orig_img_shape, padding):
        """Process raw predictions to get final bounding boxes"""
        predictions = convert_cellboxes(predictions)
        predictions = predictions.reshape(predictions.shape[0], -1, 6)  # (batch, S*S, 6)
        
        # Scale predictions back to original image size
        height, width = orig_img_shape[:2]
        pad_left, pad_top, pad_right, pad_bottom = padding
        effective_w = width + pad_left + pad_right
        effective_h = height + pad_top + pad_bottom

        predictions[..., 2] = predictions[..., 2] * effective_w - pad_left
        predictions[..., 3] = predictions[..., 3] * effective_h - pad_top
        predictions[..., 4] = predictions[..., 4] * effective_w
        predictions[..., 5] = predictions[..., 5] * effective_h
        
        # Convert predictions to list of [class_pred, prob_score, x1, y1, x2, y2]
        bboxes = []
        for box in predictions[0]:
            class_pred = int(box[0])
            prob_score = float(box[1])
            if prob_score > self.conf_threshold:
                x1 = int(box[2] - box[4]/2)
                y1 = int(box[3] - box[5]/2)
                x2 = int(box[2] + box[4]/2)
                y2 = int(box[3] + box[5]/2)
                bboxes.append([class_pred, prob_score, x1, y1, x2, y2])
        
        # Apply NMS
        return non_max_suppression(bboxes, self.nms_threshold, self.conf_threshold)
    
    def draw_boxes(self, img, boxes):
        """Draw bounding boxes and labels on the image"""
        for box in boxes:
            class_pred, prob_score, x1, y1, x2, y2 = box
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)
            
            # Draw rectangle
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{PASCAL_VOC_CLASSES[class_pred]}: {prob_score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Get label size and draw background
            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img, (x1, y1-label_h-baseline), (x1+label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(img, label, (x1, y1-baseline), font, font_scale, (0, 0, 0), thickness)
        
        return img
    
    def predict_image(self, image_path, save_path=None):
        """Predict on a single image"""
        # Load and preprocess image
        if isinstance(image_path, str):
            orig_img = cv2.imread(image_path)
            img = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            orig_img = np.array(image_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            img = image_path
        elif isinstance(image_path, np.ndarray):
            orig_img = image_path
            img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError("Input image type not supported.")
        
        input_tensor, padding = self.preprocess_image(img)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(input_tensor.to(self.device))
        
        # Process predictions
        boxes = self.process_predictions(predictions, orig_img.shape, padding)
        
        # Draw boxes
        result_img = self.draw_boxes(orig_img.copy(), boxes)
        
        # Save or return result
        if save_path:
            cv2.imwrite(save_path, result_img)
        
        return result_img, boxes
    
    def predict_video(self, video_path, output_path=None):
        """Predict on video"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process frame
            result_frame, _ = self.predict_image(frame)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Add FPS text
            fps_text = f"FPS: {1/processing_time:.2f}"
            cv2.putText(result_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame if output path is specified
            if output_path:
                out.write(result_frame)
            
            # Display progress
            frame_count += 1
            print(f"\rProcessing frame {frame_count}/{total_frames}", end="")
            
            # Display frame
            cv2.imshow('YOLOv1 Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        avg_fps = 1/np.mean(processing_times)
        print(f"\nAverage FPS: {avg_fps:.2f}")

def main():
    # Initialize predictor
    predictor = YOLOPredictor(
        checkpoint_path="checkpoint_epoch_70.pth",
        conf_threshold=0.4,
        nms_threshold=0.4
    )
    
    # Example usage for image
    image_path = "dataset/VOCdevkit/VOC2012/JPEGImages/2007_000256.jpg"
    result_img, boxes = predictor.predict_image(image_path, save_path="output.jpg")
    
    # Print detection results
    print("\nDetection Results:")
    for box in boxes:
        class_name = PASCAL_VOC_CLASSES[box[0]]
        confidence = box[1]
        print(f"{class_name}: {confidence:.2f}")
    
    # Example usage for video
    # video_path = "path/to/test/video.mp4"
    # predictor.predict_video(video_path, output_path="output.mp4")

if __name__ == "__main__":
    main()