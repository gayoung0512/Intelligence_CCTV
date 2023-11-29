from ultralytics import YOLO

# Load a model

#model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)
model = YOLO('model_structure/yolov8m.yaml').load('model/pretrained/yolov8m.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='kisa.yaml', epochs=100, imgsz=640)