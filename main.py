from ultralytics import YOLO
import os
def freeze_layer(trainer):# freeze the layers during training using callback function
    #more granular control
    #1) freeze and unfreeze layers dynamically at different points during training
    #2) freeze specific layers or groups rather than initial 'n' layers

    model = trainer.model
    num_freeze = 3 #22개의 layer 중 inital three layer
    #*YOLOv8: considerably large numver of layers*
    #*to adjust and optimize the number of layers to freeze vased on both the complexity and the nature of model*
    # freezing too many layers can cause the model to lose the ability to learn and make accurate predictions
    # try freezing only a smaller number of layers: 3~5 layers
    # retain the pre-trained weights from a previous model + fine-tune the model on a new dataset
    # freezing initial layers: focus on learning the patterns specific to dataset B
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)] # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False#no gradient should be computed for these parameters during backpropagation: freezing
            #True: unfreeze

            print(f"{num_freeze} layers are freezed.")
if __name__=='__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    dir_path=os.getcwd()
    model = YOLO("yolov8s.pt")
    model.add_callback("on_train_start", freeze_layer)
    model.train(data=dir_path+"\data\data.yaml", epochs=100, batch=32)
    #results = model.train(data='coco128.yaml', epochs=3, freeze=10) #command to freeze initial layer

    file_list=os.listdir(dir_path)

    print(dir_path)
    print(file_list)