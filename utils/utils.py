import torch
import numpy as np
from config import GRID_SIZE, NUM_CLASSES

def convert_cellboxes(predictions):
    """
    Convert model output to bounding boxes
    """
    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, GRID_SIZE, GRID_SIZE, -1)
    bboxes1 = predictions[..., 1:5]
    bboxes2 = predictions[..., 6:10]
    scores = torch.cat([predictions[..., 0:1], predictions[..., 5:6]], dim=3)
    best_box = scores.argmax(3).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(GRID_SIZE).repeat(batch_size, GRID_SIZE, 1).unsqueeze(-1)
    x = 1 / GRID_SIZE * (best_boxes[..., 0:1] + cell_indices)
    y = 1 / GRID_SIZE * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_h = 1 / GRID_SIZE * best_boxes[..., 2:4]
    
    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    predicted_class = predictions[..., 10:].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 0:1], predictions[..., 5:6])
    
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    
    return converted_preds

def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Apply Non-Max Suppression to remove overlapping boxes
    """
    assert type(bboxes) == list
    
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
            ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
        
    return bboxes_after_nms

def intersection_over_union(boxes_preds, boxes_labels):
    """
    Calculate intersection over union
    """
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)