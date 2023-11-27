from ultralytics import YOLO
import os
def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 3
    #freezing too many layers can cause the model to lose the ability to learn and make accurate predictions
    #try freezing only a smaller number of layers: 3~5 layers
    # retain the pre-trained weights from a previous model + fine-tune the model on a new dataset
    # freezing initial layers: focus on learning the patterns specific to dataset B
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)] # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
            print(f"{num_freeze} layers are freezed.")
dir_path=os.getcwd()
model = YOLO("yolov8l.pt")
model.add_callback("on_train_start", freeze_layer)
model.train(data=dir_path+"\data\data.yaml", epochs=100, batch=32)


file_list=os.listdir(dir_path)

print(dir_path)
print(file_list)