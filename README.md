# Srishti - Voice-Controlled Accessibility Assistant

Srishti is a voice-controlled accessibility assistant designed to help users operate a physical machine (likely a laser cutter or CNC).

## Features

- **Live Assistance:** Uses a webcam and a YOLO object detection model to identify and announce control panel buttons in real-time.
- **Q&A:** Answers user questions using the Gemini API (online) or a local model (offline).

## Project Structure

```
├── augmentation
│   ├── script.py
│   └── saved
├── core
│   ├── camera_handler.py
│   ├── command_handler.py
│   ├── config.py
│   ├── offline_mode.py
│   └── speech_engine.py
├── data
│   ├── knowledge_base
│   │   ├── my_data.json
│   │   ├── new_training_data.json
│   │   └── notes.txt
│   └── online_cache.json
├── offline_model_trainer
│   ├── data
│   │   ├── training_data.json
│   │   └── validation_data.json
│   ├── models
│   │   └── offline_model.pkl
│   ├── notebooks
│   │   └── data_exploration.ipynb
│   ├── src
│   │   ├── data_processing.py
│   │   ├── model_training.py
│   │   ├── offline_inference.py
│   │   ├── online_model_interface.py
│   │   └── utils.py
│   ├── config.yaml
│   ├── README.md
│   └── requirements.txt
├── project
│   ├── IMG_20250805_101742.jpg
│   ├── IMG_20250805_101751.jpg
│   ...
├── runs
│   └── detect
├── test
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── valid
│   │   ├── images
│   │   └── labels
│   ├── data.yaml
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── train.py
│   └── validate_labels.py
├── .gitignore
├── api_server.py
├── config.py
├── index.html
├── main.py
├── README.md
├── requirements.txt
├── test_mic.py
└── test_speak.py
```

## Setup

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/Srishti.git
    ```
2.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
3.  **Download Models:**
    - Download the pre-trained YOLO model (`best.pt`) and place it in the `models` folder.
    - Download the offline classification model (`offline_model.pkl`) and place it in the `offline_model_trainer/models` folder.
    > **Note:** As the models are not provided in the repository, you will need to train them yourself. Refer to the `offline_model_trainer` for more information.

## Usage

1.  **Run the application:**
    ```
    python main.py
    ```
2.  **Voice Commands:**
    - "live assistance" - Activates the live assistance mode.
    - "exit" - Exits the application.

## Known Issues

- The camera handler is inefficient and processes every frame.
- The project lacks a centralized configuration.
- The code uses a global dictionary for state management, which is not ideal.




graph TD
    %% Nodes
    User[User's Voice]
    STT[Speech-to-Text STT Engine]
    Core[Core Logic main.py]
    
    %% Vision Path
    Cam[Camera Feed]
    YOLO[Live Assistance Mode<br/>YOLO Object Detection]
    
    %% Text Path
    NetCheck{Is Internet Connected?}
    Gemini[Online Mode: Gemini API]
    Offline[Offline Logic<br/>ML Classifier/Sentence Sim]
    
    %% Output Path
    Response[Response Text_Variable]
    TTS[Text-to-Speech TTS<br/>pyttsx3]
    Speaker[Spoken Output]

    %% Connections
    User --> STT
    STT --> Core
    
    %% Logic Splitting
    Core -- "User asks to see" --> YOLO
    Cam --> YOLO
    YOLO --> Response
    
    Core -- "User asks question" --> NetCheck
    
    NetCheck -- Yes --> Gemini
    NetCheck -- No --> Offline
    
    %% Handling Gemini Errors/Safety
    Gemini -- "Success" --> Response
    Gemini -- "Error / Policy Violation" --> Offline
    
    Offline --> Response
    
    %% Final Output
    Response --> TTS
    TTS --> Speaker

## TODO

- [ ] Refactor the camera handler to be more efficient and stable.
- [ ] Implement a centralized configuration system.
- [ ] Refactor the state management to use a class.
- [ ] Create a `requirements.txt` file at the root level.
- [ ] Clean up the project structure.
- [ ] Add instructions on how to train the models.
