# 🤖 Real-Time Hand Gesture Recognition System

A sophisticated real-time hand gesture recognition system built with **MediaPipe** and **TensorFlow**, featuring live camera detection, dataset creation tools, and optimized inference capabilities.

## ✨ Features

- 🎥 **Real-time gesture detection** using webcam with live predictions
- 📊 **Interactive dataset creation mode** for collecting custom training data
- 🎬 **Video recording functionality** to capture gesture sessions
- ⚡ **TensorFlow Lite optimization** for fast inference performance
- 🖐️ **Multi-hand support** with left/right hand detection
- 📈 **Live FPS monitoring** and performance metrics
- 🎯 **Customizable gesture classes** with easy labeling system

## 🏗️ Project Structure

```
Hand Gesture/
├── 📁 data/                                    # Training datasets
│   ├── hand_landmarks_dataset.csv                 # Raw landmark data
│   └── hand_landmarks_dataset_normalized_to_the_wrist.csv
├── 📁 models/                                  # Trained models & artifacts
│   ├── gesture_classifier.keras                   # Main Keras model (recommended)
│   ├── gesture_classifier.h5                      # Legacy H5 format
│   ├── best_model.h5                              # Best training checkpoint
│   ├── gesture_classifier_savedmodel/             # TensorFlow SavedModel
│   ├── tflite/
│   │   └── gesture_classifier.tflite              # Optimized TFLite model
│   ├── gesture_classifier_architecture.json       # Model architecture
│   ├── gesture_classifier_history.json            # Training metrics
│   ├── gesture_classifier_info.json               # Model metadata
│   ├── gesture_classifier.weights.h5              # Model weights only
│   └── label_encoder.pkl                          # Class label encoder
├── 📁 notebooks/                               # Jupyter notebooks
│   ├── neural_network.ipynb                       # Model training & evaluation
│   └── data-exploration.ipynb                     # Dataset analysis & visualization
├── 📁 recordings/                              # Video recordings (auto-created)
├── 📁 temp/                                    # Temporary files & experiments
├── 🐍 main_app.py                             # Main application entry point
├── 🧠 gesture_predictor.py                    # TensorFlow Lite predictor class
├── 🛠️ utils.py                                # Utility functions & data processing
└── 📋 requirements.txt                        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam/Camera access
- 4GB+ RAM recommended

### Installation

1. **Clone and navigate to the project:**
```bash
cd "Hand Guesture"
```

2. **Create virtual environment (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python main_app.py
```

## 🎮 Usage Guide

### Real-time Gesture Recognition
- Launch the app and position your hand in front of the camera
- Gesture predictions appear in real-time with confidence scores
- Supports both left and right hand detection simultaneously

### 🎯 Interactive Controls
| Key | Action |
|-----|--------|
| `ESC` | Exit application |
| `D` | Toggle Dataset Creation Mode |
| `SPACE` | Start/Stop data collection (dataset mode) |
| `R` | Start/Stop video recording |

### 📊 Dataset Creation Workflow

1. **Enter Dataset Mode:**
   - Press `D` to activate dataset creation mode
   - Interface switches to data collection view

2. **Create Gesture Labels:**
   - Enter descriptive gesture name in console
   - Confirm label before proceeding
   - Edit labels if needed

3. **Collect Training Data:**
   - Press `SPACE` to start recording landmarks
   - Perform gesture consistently in front of camera
   - Press `SPACE` again to stop collection
   - Repeat for multiple gesture classes

4. **Data Storage:**
   - Landmarks automatically saved to `data/hand_landmarks_dataset.csv`
   - Normalized data saved to `data/hand_landmarks_dataset_normalized_to_the_wrist.csv`

## 🧠 Model Architecture & Training

### Pre-trained Models
- **gesture_classifier.keras**: Main production model (recommended)
- **gesture_classifier.tflite**: Optimized for real-time inference
- **best_model.h5**: Best checkpoint from training process

### Training Your Own Models

1. **Collect Data:** Use the dataset creation mode to gather gesture samples
2. **Open Training Notebook:** Launch `notebooks/neural_network.ipynb`
3. **Preprocess Data:** The notebook handles data loading and preprocessing
4. **Train Model:** Follow the notebook cells to train on your custom dataset
5. **Export Models:** Generate both Keras and TFLite versions

### Model Performance
- **Input:** 42 features (21 hand landmarks × 2 coordinates)
- **Processing:** Wrist-normalized coordinates for translation invariance
- **Output:** Multi-class gesture classification with confidence scores
- **Inference Speed:** ~30+ FPS on modern hardware with TFLite

## 🔧 Technical Details

### Core Components

**main_app.py**
- Camera interface and real-time processing
- Dataset creation workflow management
- Video recording functionality
- Interactive UI with live feedback

**gesture_predictor.py**
- TensorFlow Lite model wrapper
- Fast inference optimization
- Confidence scoring and class prediction

**utils.py**
- MediaPipe landmark extraction
- Data preprocessing and normalization
- CSV dataset management
- Feature engineering utilities

### Data Processing Pipeline
1. **MediaPipe Detection:** Extract 21 hand landmarks per frame
2. **Coordinate Normalization:** Relative to wrist position for invariance
3. **Feature Vector:** 42-dimensional input (x,y coordinates)
4. **Model Inference:** TensorFlow Lite optimized prediction
5. **Post-processing:** Confidence thresholding and smoothing

## 📈 Performance Optimization

- **TensorFlow Lite:** 3-5x faster inference than standard TensorFlow
- **Frame Skipping:** Configurable processing intervals for performance tuning
- **Efficient Rendering:** Optimized OpenCV drawing operations
- **Memory Management:** Proper resource cleanup and memory handling

## 🛠️ Development & Customization

### Adding New Gestures
1. Use dataset creation mode to collect samples
2. Retrain model using the provided notebook
3. Update class labels and model files
4. Test with real-time prediction

### Model Improvements
- Experiment with different architectures in the notebook
- Adjust hyperparameters for better accuracy
- Implement data augmentation techniques
- Add temporal smoothing for stability

## 📋 Dependencies

Key libraries and their purposes:
- **mediapipe**: Hand landmark detection
- **tensorflow**: Model training and inference
- **opencv-python**: Camera interface and image processing
- **numpy**: Numerical computations
- **scikit-learn**: Data preprocessing and evaluation
- **pandas**: Dataset manipulation
- **matplotlib/seaborn**: Visualization and analysis

## 🤝 Contributing

Feel free to contribute by:
- Adding new gesture classes
- Improving model accuracy
- Optimizing performance
- Enhancing the user interface
- Adding new features

## 📄 License

This project is available under a **dual licensing model**:

- **🆓 Non-Commercial Use**: Free for educational, research, and personal projects
- **💼 Commercial Use**: Requires a paid commercial license

**Commercial use includes:**
- Integration into commercial products or services
- Use in revenue-generating applications
- Distribution as part of commercial software
- Business operations or commercial research

For commercial licensing inquiries, please contact the project maintainer.

See [LICENSE](LICENSE) for full terms and conditions.

---

**Ready to recognize gestures in real-time? Run `python main_app.py` and start creating your custom gesture dataset!** 🚀