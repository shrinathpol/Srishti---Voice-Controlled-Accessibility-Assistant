from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Load a pre-trained base model
    model = YOLO('yolo12l.pt') 

    # Train the model using your custom dataset
    print("Starting model training on GPU...")
    results = model.train(
       data='test/data.yaml',
       epochs=100,
       imgsz=640,
       project='runs/detect',
       device=[0],  # <-- Add this line to specify the first GPU (GPU:0)
       workers=4,
       batch=8
    )

    print("\nTraining complete!")
    print("Your trained model 'best.pt' is in the 'runs/detect/train/weights/' folder.")

