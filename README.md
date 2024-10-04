# FaceRecognition

## Overview

The **FaceRecognition** project is a Python application designed to recognize facial emotions from images using machine learning techniques. It uses Singular Value Decomposition (SVD) to train on a dataset of faces. The project also includes real-time face recognition using a webcam.

## Features

- **Image Preprocessing**: Converts images to grayscale and flattens them for analysis.
- **Training Model**: Utilizes Singular Value Decomposition (SVD) to project images into a lower-dimensional space.
- **Face Recognition**: Compares new images to a training set and identifies the closest match based on Euclidean distance.
- **Real-time Recognition**: Uses a webcam to detect and recognize faces in real-time, displaying the recognized emotion.

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/YourUsername/FaceRecognition
```

### Navigate to the Project Directory

```bash
cd FaceRecognition
```

### Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

```bash
pip install numpy opencv-python
```

### 1. Create the Training and Testing Datasets

#### Dataset Structure

```
Dataset/
├── test                  # Validation set
│   ├── angry
│   ├── disgust
│   ├── fear
│   ├── happy
│   ├── neutral
│   ├── sad
│   └── surprise
└── train                 # Training set
    ├── angry
    ├── disgust
    ├── fear
    ├── happy
    ├── neutral
    ├── sad
    └── surprise
```

Each subfolder in both `train` and `test` directories should contain the corresponding facial images for the emotions. The structure will look like this:

### 2. Prepare the Dataset

Place the images you want to use for training and testing inside the respective `train` and `test` subfolders. These should be grayscale images of faces expressing emotions like `angry`, `happy`, `sad`, etc.

## Run the Application

### To Train the Model and Recognize Faces from the Webcam

Run the following command:

```bash
python EmotionDetector.py
```

- The application will open your webcam, detect faces, and display the recognized emotion in real-time.
- It will compare the live face frames with the trained model and show the corresponding emotion on the screen.
- Press `q` to quit the application.

## Output

- The application will display a video feed of the camera with rectangles drawn around detected faces.
- The recognized emotion will be displayed above each face.

### Files in the Directory

- **EmotionDetector.py**: Main script to run the application and detect emotions in real-time using the webcam.
- **README.md**: Project documentation.
- **haarcascade_frontalface_default.xml**: The pre-trained OpenCV model for face detection.

## Contribution Guidelines

- **Reporting Issues**: Please report any issues or bugs using the GitHub Issues tab.
- **Submitting Pull Requests**: Fork the repository, make your changes, and submit a pull request with a description of your changes.

## Related Work

For further reading, you can refer to the foundational paper that inspired this project: [Eigenfaces for Recognition](https://sites.cs.ucsb.edu/~mturk/Papers/mturk-CVPR91.pdf).

You can also refer to my previous Project: [Face Recognition using eigen faces](https://github.com/AdithyaVarma28/Face-recognition-using-eigenfaces/tree/main).

## Directory Structure

```
.
├── Dataset
│   ├── test
│   │   ├── angry
│   │   ├── disgust
│   │   ├── fear
│   │   ├── happy
│   │   ├── neutral
│   │   ├── sad
│   │   └── surprise
│   └── train
│       ├── angry
│       ├── disgust
│       ├── fear
│       ├── happy
│       ├── neutral
│       ├── sad
│       └── surprise
├── EmotionDetector.py
├── README.md
└── haarcascade_frontalface_default.xml
```