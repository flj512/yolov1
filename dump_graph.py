import torch
from torch.utils.tensorboard import SummaryWriter

from models.yolo import YOLOv1

model = YOLOv1(grid_size=7, num_boxes=2, num_classes=20, pretrained=True)
writer = SummaryWriter("runs/yolo_v1")
writer.add_graph(model, torch.rand(1, 3, 448, 448))
writer.close()
