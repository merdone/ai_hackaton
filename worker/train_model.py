from ultralytics import YOLO

def main():
    model = YOLO('../models/yolov8l.pt')

    results = model.train(
        data='C:/Users/maksn/PycharmProjects/ai_hackaton/datasets/worker_data/dataset.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        batch=8,
        workers=4,
        project='runs/worker_detect',
        name='v1_experiment'
    )

if __name__ == '__main__':
    main()