from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("D:\\Python\\traking\\runs\\detect\\train\\weights\\best.pt")


# Train the model on the COCO8 example dataset for 100 epochs
# results = model(source="0", show=True, conf=0.4, save=True)
results = model(source="0", show=True, conf=0.4, save=True, stream=True)

