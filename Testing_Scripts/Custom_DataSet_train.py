if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()

    from ultralytics import YOLO

# # Load a model
    model = YOLO('../YOLO_Weights/yolov8n.pt')  # load a pretrained model (recommended for training)

# # Train the model with 2 GPUs
    results = model.train(data="D:\Pycharm_projects\SmartPark Tracker\Train_dataset\data.yaml",imgsz=640)


