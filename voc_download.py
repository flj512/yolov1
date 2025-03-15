import torchvision

# Load Pascal VOC 2012
voc_dataset = torchvision.datasets.VOCDetection(
    root="dataset", year="2012", image_set="train", download=True
)
