import torch
import torch.nn as nn
from config import LAMBDA_COORD, LAMBDA_NOOBJ

class YOLOLoss(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(YOLOLoss, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
    def compute_iou(self, box1, box2):
        """
        Compute IOU between box1 and box2
        box1, box2: [x, y, w, h]
        """
        # Convert to x1, y1, x2, y2
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
        
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
        
        # Intersection
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        
        intersection = torch.clamp((x2 - x1), 0) * torch.clamp((y2 - y1), 0)
        
        # Union
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)
    
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        
        # Reshape predictions
        predictions = predictions.view(batch_size, self.grid_size, self.grid_size, -1)
        
        # Get predictions for both boxes
        box1_pred = predictions[..., :5]  # First box
        box2_pred = predictions[..., 5:10]  # Second box
        class_pred = predictions[..., 10:]  # Class predictions
        
        # Get target values
        box_target = targets[..., :4]
        obj_target = targets[..., 4:5]
        class_target = targets[..., 5:]
        
        # Calculate IOUs for both predicted boxes
        iou_box1 = self.compute_iou(box1_pred[..., :4], box_target)
        iou_box2 = self.compute_iou(box2_pred[..., :4], box_target)
        ious = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        
        # Get the box with highest IOU
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = obj_target[..., 0].unsqueeze(3)  # Identity of object i in cell i
        
        # ======================== #
        #   FOR BOX COORDINATES   #
        # ======================== #
        
        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest IOU calculated previously
        box_predictions = exists_box * (
            (best_box * box2_pred + (1 - best_box) * box1_pred)
        )
        
        box_targets = exists_box * box_target
        
        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions[..., :4], end_dim=-2),
            torch.flatten(box_targets[..., :4], end_dim=-2),
            batch_size
        )
        
        # ==================== #
        #   FOR OBJECT LOSS   #
        # ==================== #
        
        pred_box = (best_box * box2_pred[..., 4:5] + (1 - best_box) * box1_pred[..., 4:5])
        
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * obj_target[..., 0:1]),
            batch_size
        )
        
        # ======================= #
        #   FOR NO OBJECT LOSS   #
        # ======================= #
        
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * box1_pred[..., 4:5]),
            torch.flatten((1 - exists_box) * obj_target[..., 0:1]),
            batch_size
        )
        
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * box2_pred[..., 4:5]),
            torch.flatten((1 - exists_box) * obj_target[..., 0:1]),
            batch_size
        )
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        
        # (N, S, S, 20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * class_pred, end_dim=-2),
            torch.flatten(exists_box * class_target, end_dim=-2),
            batch_size
        )
        
        # ================== #
        #   TOTAL LOSS      #
        # ================== #
        l1 = LAMBDA_COORD * box_loss # First two rows of paper
        l2 = object_loss # Third row
        l3 = LAMBDA_NOOBJ * no_object_loss  # Fourth row
        l4 = class_loss # Fifth row

        loss = l1 + l2 + l3 + l4
        
        loss_ratio = torch.tensor([l1.item(), l2.item(), l3.item(), l4.item()])/(loss.item() + 1e-6)

        return loss, loss_ratio
    
    def mse(self, pred, target, batch_size):
        return torch.sum((pred - target) ** 2)/batch_size